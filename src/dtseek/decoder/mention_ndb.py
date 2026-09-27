"""提及检索记忆（Mention-NDB）—— 把 nanoSeek 的 NDB 映射到 DTSeek 的切片决策。

nanoSeek 的 NDB（`nanoSeek/model/ngram_ndb.py`，v7）是「**后缀 n-gram → 下一个 token
的分布**」的只读表 + 可学习门控：表是 no_grad 的数据统计量，模型学的是「写多重 / 信多少」。

DTSeek 不生成 token，它每步输出 `(类别, 起点, 终点)`。所以不能照搬 token 续写，
本文件做的是下面这个映射（**这是移植的关键设计，不是机械搬运**）：

    nanoSeek NDB                      Mention-NDB（本文件）
    ─────────────────────────────     ─────────────────────────────────────────
    键：token 后缀哈希（全局语料）      键：**当前文档内**提及起始处的 token n-gram 哈希
    值：下一个 token 的 top-K 分布      值：该键此前被分配过的**身份槽 id** 的加权计数
    写：σ(W_w·h) 加权计 token 计数      写：σ(W_w·h) 加权计 (键 → 真值 label) 计数
    读：查表 → p_ng，与 p_model 凸混合   读：按 start 指针分布聚合查表 → p_ng，与 p_cls 凸混合
    多级：后缀长度 (8,6) 混合            多级：n-gram 长度 (1,2) 混合（首个字 / 双字前缀）
    表：全局、一个 run 建一次             表：**每个 batch 一张**，逐样本各占一行

## 为什么表必须是「每样本一张」（与 nanoSeek 最大的不同）

person 卡的 label 是**匿名身份槽**，按段落内首次出场顺序分配 —— 同一个「张伟」在样本 A
里是人物1、在样本 B 里是人物3。所以一张跨样本的全局「字面 → id」表只能学到 id 的边缘
分布（≈ 均匀），等于没学。真正要检索的是「**这篇文本里**，这个字面刚才被指成了谁」——
即一段**情节记忆（episodic memory）**，随文档开始清零、随提及逐个写入。

这也正是 person 卡当前失败的机制：`repeat_mention_acc` 36% 不是容量问题（层数减半指标
几乎不变），而是解码头没有任何地方存「我已见过谁」——query 序列里只有上一步的
(label, 位置)，拿不到更早那次提及的绑定。

## 硬约束（都在代码里强制）

1. **不碰全局指针**：读路径只接收 `start_logits`（detached 后当权重用），start/end 指针
   的计算一字未动 —— 既不改数值，也不引入新梯度。`tests/test_mention_ndb.py` 有断言。
2. **表有硬上限**：构造 / 每次 `reset` 都按 `max_table_gb` 拦截，超了就抛错，不 OOM 到一半。
3. **表是 no_grad 的**（数据统计量，不是参数）：只有 `write_gate` / `read_gate` /
   `level_weight` 三个门控是 nn.Parameter。梯度到 `write_gate` 的路径与 nanoSeek 相同：
   **不是通过写**（离散查表阻断了跨步梯度），而是把写门控值同时喂进读门控的输入。

## 写入必须显式开启

`read()` 永远安全；`write()` 只在 `write_enabled()` 里生效。训练循环只把**训练 micro-batch**
包进去，验证 / 探针绝不包 —— 否则验证集会写进记忆（nanoSeek 被 val 泄漏坑过）。

## 步时代价：文档级预计算 + 每步 O(1) gather/scatter

规格是「每篇文档预计算一次键 + 每步 O(1) gather/scatter」。落地成三件事（数值逐位不变）：

1. **键只算一次**：n-gram 哈希与取模原本在每个 decode step 重算 O(L) 一遍（16 步 × 2 级），
   现在 `_ensure_flat()` 在每批第一次用到 `input_ids` 时算成 `_flat[B, L, nlev]`，
   之后每步只是把它按位置 gather 出来。**同一批内 `input_ids` 必须是同一个张量对象**
   （`reset()` 会作废缓存，所有调用点都是每批一次 `reset`）。
2. **两级合并成一张表**：`(counts, totals)` 从「每级一张 [B, slots_li, C]」变成
   「一张 [B, Σslots, C]/[B, Σslots]」，级间用偏移区分。好处是写只需**两次**
   `index_add_`（而不是每级各两次），而不是靠 Python 层循环。
3. **读权重是离散的（one-hot）**：`read_attention(hard=True)` 与教师强制下的真值起点都产生
   one-hot 注意力。此时 `Σ_L α_L·frac_L` 塌缩成「在**那一个位置**取值」——
   `read()` 认出这次 `attn` 就是自己刚发出去的那个对象（`attn is self._fast_attn`），
   走单点 gather；否则（软读诊断 / 手工构造的 attn）回退到原来的稠密路径。
   两条路径的算式逐项对应，单测与 `scripts/ndb_bench.py --mode check` 逐位比对。

另外两类**纯诊断**开销被改成惰性物化（数值不变，只是不再每步同步）：
`read()` 里的 `last_gate / last_covered` 与 `write()` 里的 `n_written / 返回值`。
原先每次读写都 `.item()/float()`，等于每个训练步 60+ 次设备同步。
"""
from __future__ import annotations

import contextlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["MentionNDB"]

#: 多项式滚动哈希的乘子，与 nanoSeek `suffix_hashes` 同源（1000003）。
_HASH_MULT = 1_000_003
#: 支持的最大 n-gram 级数。int64 下 4 阶会溢出（8192^4），所以硬拦在 3。
_MAX_LEVEL = 3
#: `f_total` 的归一化分母，与原始实现一致（`log1p(100)`）。
_LOG1P_100 = math.log1p(100.0)


class MentionNDB(nn.Module):
    """提及 span 的情节检索记忆：no_grad 表 + 可学习读写门控。

    参数
    ----
    hidden_dim   : 门控网络输入维度（= 解码头隐维）
    num_classes  : 身份槽数（含背景类 0）
    vocab_size   : 字表大小（键的取值域）
    levels       : n-gram 阶数列表，如 `(1, 2)`
    slots        : 每级槽位数（int 或与 levels 等长）
    max_table_gb : 表显存上限，超限抛错
    read_true    : True = 训练时用**真值起点** one-hot 做读注意力（教师强制）；
                   False = 训练与推理都用 start 指针的 softmax（部署口径一致）。
    """

    def __init__(self, hidden_dim: int, num_classes: int, vocab_size: int,
                 levels: tuple[int, ...] = (1, 2), slots=(8192, 4096),
                 max_table_gb: float = 0.25, read_true: bool = True):
        super().__init__()
        self.levels = tuple(int(x) for x in levels)
        if not self.levels:
            raise ValueError("levels 不能为空")
        if max(self.levels) > _MAX_LEVEL:
            raise ValueError(f"levels 最大 {_MAX_LEVEL}（int64 哈希再高会溢出），收到 {self.levels}")
        if isinstance(slots, int):
            slots = [slots] * len(self.levels)
        self.slots = [int(s) for s in slots]
        if len(self.slots) != len(self.levels):
            raise ValueError("slots 与 levels 长度必须一致")
        if any(s < 1 for s in self.slots):
            raise ValueError(f"slots 必须 >= 1，收到 {self.slots}")

        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.vocab_size = int(vocab_size)
        self.max_table_gb = float(max_table_gb)
        self.read_true = bool(read_true)
        self.n_stats = 3        # [log(total), top1 占比, 覆盖质量]
        self.n_levels = len(self.levels)

        # 级间偏移：两级共用一张 [B, Σslots, C] 表，级 li 的槽位落在 [off, off+slots[li])
        offs, acc = [], 0
        for s in self.slots:
            offs.append(acc)
            acc += s
        self._offsets = offs
        self._total_slots = acc

        # ---- 表（**故意不注册成 buffer**）：no_grad、不进优化器、不进 state_dict ----
        self._tab_cnt: torch.Tensor | None = None     # [B, Σslots, C]
        self._tab_tot: torch.Tensor | None = None     # [B, Σslots]
        self._batch = 0
        self._write_on = False
        # 文档级键缓存：[B, L, nlev] 的**扁平槽位**（已含级间偏移）+ 它对应的 input_ids 对象
        self._flat: torch.Tensor | None = None
        self._flat_src: torch.Tensor | None = None
        # 最近一次 read_attention 的 one-hot 元数据（仅当它确实是离散的时候有效）
        self._fast_attn: torch.Tensor | None = None
        self._fast_pos: torch.Tensor | None = None
        self._fast_row: torch.Tensor | None = None
        # 纯诊断量：**设备上的标量**，只在 stats() 里物化成 Python 数（避免每步同步）
        self._n_written_t: torch.Tensor | None = None
        self._last_gate_t: torch.Tensor | None = None
        self._last_covered_t: torch.Tensor | None = None
        self._ret_ok_t: torch.Tensor | None = None
        self._ret_n_t: torch.Tensor | None = None
        self._ret_cover_t: torch.Tensor | None = None
        self._ret_tot_t: torch.Tensor | None = None

        # ---- 可学习门控（**只有这些是参数**）----
        self.write_gate = nn.Linear(self.hidden_dim, 1)
        nn.init.zeros_(self.write_gate.weight)
        nn.init.constant_(self.write_gate.bias, 1.0)      # σ(1)≈0.73：起点先平均地写
        # 读门控**不能全零**初始化：写门控的梯度要经它回流（同 nanoSeek 的死启动坑）。
        self.read_gate = nn.Linear(self.hidden_dim + self.n_stats + 1, 1)
        nn.init.constant_(self.read_gate.bias, -2.0)      # 初始 ≈0.12：先轻信
        self.level_weight = nn.Parameter(torch.zeros(len(self.levels)))

    # ==================================================================
    # 监控量：设备标量 → Python 数（**只在真正读的时候同步一次**）
    # ==================================================================
    @property
    def n_written(self) -> int:
        """累计写入的 (样本, 提及) 数。累加在设备上，读取时才同步。"""
        return 0 if self._n_written_t is None else int(self._n_written_t.item())

    def _scalar(self, t: torch.Tensor | None) -> float:
        return float("nan") if t is None else float(t.item())

    def _int_scalar(self, t: torch.Tensor | None) -> int:
        return 0 if t is None else int(t.item())

    @property
    def _counts(self) -> list[torch.Tensor]:
        """按级切出的表视图（**只读**，供观测 / 对照用；写入走合并表的扁平索引）。"""
        if self._tab_cnt is None:
            return []
        return [self._tab_cnt[:, self._offsets[li]:self._offsets[li] + self.slots[li], :]
                for li in range(self.n_levels)]

    @property
    def _totals(self) -> list[torch.Tensor]:
        if self._tab_tot is None:
            return []
        return [self._tab_tot[:, self._offsets[li]:self._offsets[li] + self.slots[li]]
                for li in range(self.n_levels)]

    # ==================================================================
    # 表生命周期
    # ==================================================================
    def table_bytes(self, batch_size: int) -> int:
        """表占用字节数 = 全部级的 slots × (counts 的 num_classes 项 + totals 1 项) × 4B。"""
        return int(batch_size) * (self.num_classes + 1) * sum(self.slots) * 4

    def table_gb(self, batch_size: int | None = None) -> float:
        return self.table_bytes(batch_size or self._batch) / 2**30

    def reset(self, batch_size: int, device) -> None:
        """为一批新样本开一张空表（**逐样本的情节记忆**，每批重建）。"""
        need = self.table_bytes(batch_size) / 2**30
        if need > self.max_table_gb:
            raise ValueError(
                f"NDB 表需要 {need:.3f}GB（batch={batch_size}, slots={self.slots}, "
                f"classes={self.num_classes}），超过上限 {self.max_table_gb}GB。"
                " 缩小 slots / batch，或提高 max_table_gb。")
        self._tab_cnt = torch.zeros(batch_size, self._total_slots, self.num_classes, device=device)
        self._tab_tot = torch.zeros(batch_size, self._total_slots, device=device)
        self._batch = int(batch_size)
        # 新一批 → 文档级键缓存作废（input_ids 换了）
        self._flat = None
        self._flat_src = None
        self._fast_attn = self._fast_pos = self._fast_row = None

    def reset_stats(self) -> None:
        """清空跨 batch 累计的诊断计数（`reset()` 不会动它们 —— 那些是要跨 batch 求和的）。"""
        self._n_written_t = None
        self._ret_ok_t = self._ret_n_t = self._ret_cover_t = self._ret_tot_t = None

    @contextlib.contextmanager
    def write_enabled(self):
        """只有包在这里面的 `write()` 才真正落表（防 val 泄漏的显式开关）。"""
        prev, self._write_on = self._write_on, True
        try:
            yield self
        finally:
            self._write_on = prev

    @property
    def write_active(self) -> bool:
        return self._write_on

    # ==================================================================
    # 键
    # ==================================================================
    def _keys(self, input_ids: torch.Tensor, pos: torch.Tensor, n: int) -> torch.Tensor:
        """由 `input_ids[B, L]` 在位置 `pos[B, P]` 处取 n-gram 的多项式滚动哈希。

        对齐约定：位置 p 的键由 `tokens[p : p+n]` 组成（**向前**取，不是向后）——
        提及起始处的字面前缀。`p+n` 越界时夹到最后一个位置（末尾单字提及的已知近似）。
        """
        last = input_ids.shape[1] - 1
        key = input_ids.gather(1, pos.clamp(0, last)).to(torch.int64)
        for j in range(1, n):
            pj = (pos + j).clamp(0, last)
            key = key * _HASH_MULT + input_ids.gather(1, pj).to(torch.int64)
        return key

    def _slots_at(self, input_ids: torch.Tensor, pos: torch.Tensor, li: int) -> torch.Tensor:
        key = self._keys(input_ids, pos, self.levels[li])
        return torch.remainder(key, self.slots[li])

    def _ensure_flat(self, input_ids: torch.Tensor) -> torch.Tensor:
        """把「每个位置 × 每一级」的扁平槽位算一次并缓存（**文档级预计算**）。

        缓存按 `input_ids` 的**对象身份**作废：同一批里 read/write 都传同一个张量
        （`runtime.task_loss` / `evaluate_task` / `_rollout` 都是先取一次再复用），
        `reset()` 换批时显式作废。传进来别的对象就重算 —— 只会慢，不会错。
        """
        if self._flat is not None and self._flat_src is input_ids:
            return self._flat
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        cols = [torch.remainder(self._keys(input_ids, pos, n), self.slots[li]) + self._offsets[li]
                for li, n in enumerate(self.levels)]
        self._flat = torch.stack(cols, dim=-1)      # [B, L, nlev]
        self._flat_src = input_ids
        return self._flat

    # ==================================================================
    # 写
    # ==================================================================
    def write(self, h: torch.Tensor, input_ids: torch.Tensor, starts: torch.Tensor,
              labels: torch.Tensor, valid: torch.Tensor) -> None:
        """把本步「提及起始字面 → 真值身份槽」写进**本样本自己**的表。

        h      : [B, D] 本步隐状态（只用来算写门控权重，不作为检索键）
        starts : [B]   提及起始位置（训练时 = 真值；rollout 时 = 模型自己预测的）
        labels : [B]   身份槽 id（0 = 背景，不写）
        valid  : [B]   该步是否有效（step_mask；0 的样本不写）

        返回 `None`（写入是否开启的判据是返回值 is None，与旧行为一致）。旧版还返回
        `float((w*ok).sum())` 作为诊断标量，但**没有任何调用点消费它**，却强制每步一次
        设备同步；计数改由 `n_written` 累加在设备上，读取时才物化。
        """
        if not self._write_on:
            return None
        with torch.no_grad():
            B = h.shape[0]
            w = torch.sigmoid(self.write_gate(h)).squeeze(-1) * valid.float()
            ok = (valid > 0.5) & (labels > 0)
            flat_slots = self._ensure_flat(input_ids)
            ar = torch.arange(B, device=h.device)
            inc = torch.zeros(B, self.num_classes, device=h.device)
            inc[ar, labels.clamp(0, self.num_classes - 1)] = (w * ok).to(inc.dtype)
            # 一次 gather 拿到「本步位置」在各级的扁平槽位，再一次 index_add_ 写全表。
            # 同一批次内 B×nlev 个扁平索引互不相同（不同样本/不同级偏移），
            # 所以 index_add_ 的累加顺序无歧义 —— 与旧版逐级 index_add_ 逐位一致。
            # `starts` 必须 clamp 到 [0, L-1]：旧 `_keys` 就是这么做的，而 `engine.py`
            # 会传 `min(len(segment_text)-1, ...)`（可能越出 padded 窗口右端）。
            idx = flat_slots.gather(1, starts.clamp(0, input_ids.shape[1] - 1).view(B, 1, 1)
                                    .expand(B, 1, self.n_levels)).squeeze(1)      # [B, nlev]
            off = (ar.unsqueeze(1) * self._total_slots + idx).reshape(-1)
            self._tab_cnt.view(B * self._total_slots, self.num_classes).index_add_(
                0, off, inc.unsqueeze(1).expand(B, self.n_levels, self.num_classes)
                .reshape(B * self.n_levels, self.num_classes))
            self._tab_tot.view(-1).index_add_(
                0, off, (w * ok).unsqueeze(1).expand(B, self.n_levels).reshape(-1))
            if self._n_written_t is None:
                self._n_written_t = torch.zeros((), dtype=torch.long, device=h.device)
            self._n_written_t = self._n_written_t + ok.sum()
        return None

    # ==================================================================
    # 读
    # ==================================================================
    def read(self, cls_logits: torch.Tensor, h: torch.Tensor, input_ids: torch.Tensor,
             attn: torch.Tensor, targets: torch.Tensor | None = None) -> torch.Tensor:
        """按位置注意力聚合查表，把检索到的身份分布与分类头凸混合。

        返回**混好之后的 log-probabilities**（形状同 `cls_logits`）：
        `p = (1-g)·softmax(cls_logits) + g·Σ_L α_L·p_ng^(L)`，再取 log。
        用 log 概率当「logits」喂 `cross_entropy` 是等价的（log_softmax 幂等），
        `argmax` 也不变，所以调用方无需区分。

        attn    : [B, L] 位置权重（已 detach、已按 padding 归零、已归一化）
        targets : [B] 可选的诊断真值（**负值 = 该步无效，跳过**）。传了才统计「**纯检索**（不含分类头）在覆盖到的
                  样本上 top-1 命中率」——这是判断「记忆里到底有没有那个答案」的唯一口径，
                  与最终预测混在一起会分不清是检索没用还是门控没开。

        `attn` 恰好是刚由 `read_attention()` 返回的那个 one-hot 张量时，走 O(1) 单点
        gather 路径（见模块 docstring）；否则回退到稠密路径，两条路径数值逐位相同。
        """
        if self._tab_cnt is None:
            raise RuntimeError("read() 前必须先 reset(batch_size, device)")
        B, L = input_ids.shape
        dev = cls_logits.device
        alpha = torch.softmax(self.level_weight, dim=0)
        flat_slots = self._ensure_flat(input_ids)
        if attn is self._fast_attn and self._fast_attn is not None:
            p_ng, feat, covered_any = self._gather_onehot(flat_slots, attn, alpha, B, dev)
        else:
            p_ng, feat, covered_any = self._gather_dense(flat_slots, attn, alpha, B, L, dev)

        p_ng = p_ng / p_ng.sum(-1, keepdim=True).clamp_min(1e-9)
        covered = (covered_any > 0).to(cls_logits.dtype).unsqueeze(-1)

        w_t = torch.sigmoid(self.write_gate(h)).squeeze(-1)
        g = torch.sigmoid(self.read_gate(torch.cat([h, feat, w_t.unsqueeze(-1)], dim=-1)))
        # 槽为空的位置 p_ng ≡ 0：门控必须归零，否则会把 p_model 整体缩小（凸组合被破坏）
        g = g * covered
        p_model = F.softmax(cls_logits.float(), dim=-1)
        p_new = (1 - g) * p_model + g * p_ng
        p_new = p_new.clamp_min(1e-9)
        p_new = p_new / p_new.sum(-1, keepdim=True)

        with torch.no_grad():
            # 只记设备标量，不在这一步物化（旧版每步两次 float() 同步）
            self._last_gate_t = g.mean()
            self._last_covered_t = covered.mean()
            if targets is not None:
                sel = targets >= 0            # 负值 = 该步无效（与 ignore_index 同约定）
                cov = (covered_any > 0) & sel
                hit = (p_ng.argmax(-1) == targets) & cov
                self._accum(self._ret_ok_t, hit.sum(), "_ret_ok_t")
                self._accum(self._ret_n_t, cov.sum(), "_ret_n_t")
                self._accum(self._ret_cover_t, cov.sum().to(p_ng.dtype), "_ret_cover_t")
                self._accum(self._ret_tot_t, sel.sum(), "_ret_tot_t")
        return torch.log(p_new).to(cls_logits.dtype)

    def _accum(self, cur: torch.Tensor | None, add: torch.Tensor, name: str) -> None:
        setattr(self, name, add if cur is None else cur + add)

    def _gather_onehot(self, flat_slots: torch.Tensor, attn: torch.Tensor,
                       alpha: torch.Tensor, B: int, dev) -> tuple:
        """单点路径：attn 是 one-hot，`Σ_L α_L·frac_L` 塌缩成在该位置取值。

        逐项对应稠密路径（`frac/p_l/mass/tot_agg/f_share` 的定义完全一致），
        `row = attn[b, pos]` 就是那一行的和（one-hot → 恰为 0.0 / 1.0），
        于是 `(attn·X).sum(1) == X[pos]·row` 逐位成立。
        """
        pos = self._fast_pos
        row = self._fast_row                                   # [B]，0.0 / 1.0
        idx = flat_slots.gather(1, pos.view(B, 1, 1)
                                .expand(B, 1, self.n_levels)).squeeze(1)      # [B, nlev]
        cnt = self._tab_cnt.gather(1, idx.unsqueeze(-1).expand(B, self.n_levels, self.num_classes))
        tot = self._tab_tot.gather(1, idx)                      # [B, nlev]
        cov = (tot > 0).to(cnt.dtype)
        frac = cnt / tot.clamp_min(1e-6).unsqueeze(-1) * cov.unsqueeze(-1)
        p_l = frac * row.view(B, 1, 1)
        # 级内重新归一：top-K / 哈希碰撞下 `frac` 未必是分布（同 nanoSeek 的注释）
        p_l = p_l / p_l.sum(-1, keepdim=True).clamp_min(1e-9)
        mass = cov * row.view(B, 1)                             # [B, nlev] 覆盖质量
        tot_agg = tot * row.view(B, 1)
        f_total = torch.log1p(tot_agg) / _LOG1P_100
        f_share = p_l.max(-1).values
        feats = [torch.stack([f_total[:, li], f_share[:, li], mass[:, li]], dim=-1)
                 for li in range(self.n_levels)]
        # 与稠密路径同序累加（0 + α₀p₀ = α₀p₀ 在浮点下逐位成立，p 非负）
        p_ng = alpha[0] * p_l[:, 0]
        covered_any = mass[:, 0]
        for li in range(1, self.n_levels):
            p_ng = p_ng + alpha[li] * p_l[:, li]
            covered_any = covered_any + mass[:, li]
        feat = torch.stack(feats, 0).mean(0)                    # [B, 3]
        return p_ng, feat, covered_any

    def _gather_dense(self, flat_slots: torch.Tensor, attn: torch.Tensor,
                      alpha: torch.Tensor, B: int, L: int, dev) -> tuple:
        """稠密路径：任意 `attn`（软读诊断 / 手工构造）。语义与旧实现逐位一致。"""
        p_ng = torch.zeros(B, self.num_classes, device=dev)
        covered_any = torch.zeros(B, device=dev)
        feats = []
        for li in range(self.n_levels):
            slot = flat_slots[:, :, li]                         # [B, L] 扁平槽位
            cnt = self._tab_cnt.gather(1, slot.unsqueeze(-1).expand(B, L, self.num_classes))
            tot = self._tab_tot.gather(1, slot)                 # [B, L]
            cov = (tot > 0).to(cnt.dtype)
            frac = cnt / tot.clamp_min(1e-6).unsqueeze(-1) * cov.unsqueeze(-1)
            a = attn.unsqueeze(-1)
            p_l = (a * frac).sum(1)
            p_l = p_l / p_l.sum(-1, keepdim=True).clamp_min(1e-9)
            mass = (attn * cov).sum(1)
            covered_any = covered_any + mass
            p_ng = p_ng + alpha[li] * p_l
            tot_agg = (attn * tot).sum(1)
            f_total = torch.log1p(tot_agg) / _LOG1P_100
            f_share = p_l.max(-1).values
            feats.append(torch.stack([f_total, f_share, mass], dim=-1))
        feat = torch.stack(feats, 0).mean(0)                    # [B, 3]
        return p_ng, feat, covered_any

    # ==================================================================
    # 监控
    # ==================================================================
    def read_attention(self, start_logits: torch.Tensor, mask: torch.Tensor,
                       true_starts: torch.Tensor | None = None,
                       hard: bool = True) -> torch.Tensor:
        """构造读注意力：`read_true` 且给了真值起点 → one-hot；否则由 start 指针决定。

        `hard=True`（默认）：取 `argmax` 的 one-hot。**必须是硬的** —— 软注意力把权重
        摊到很多位置上，一个「表里没有的新人物」也会因为某些位置有键而被判成「已覆盖」，
        读门控于是打开并塞一个**已有的 id** 进去。实测：软读把首次提及正确率从 0.999
        打到 0.480（`scripts/ndb_probe.py` 的 ndb_pred / soft 对照）。键是离散的，
        读权重也必须离散。

        `hard=False` 只留给诊断（复现上面那次失败）。

        两种都 **detach**：start/end 指针的数值与梯度都不受影响（硬约束 1）。

        返回值同时被登记为「本步的可信 one-hot」：`read()` 只有在收到**同一个对象**时
        才走单点路径（软读、手工构造的 attn 一律回退稠密路径）。
        """
        B, L = start_logits.shape
        with torch.no_grad():
            if self.read_true and true_starts is not None:
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
            if pos is not None:
                # one-hot 行的和恰为 0.0/1.0，等价于 (attn·X).sum(-1) 里的那个行权重
                row = a.gather(1, pos.view(B, 1)).squeeze(1)
                self._fast_attn, self._fast_pos, self._fast_row = a, pos, row
            else:
                self._fast_attn = self._fast_pos = self._fast_row = None
        return a

    def stats(self) -> dict:
        ret_n = self._int_scalar(self._ret_n_t)
        ret_tot = self._int_scalar(self._ret_tot_t)
        return {
            "n_written": self.n_written,
            "table_gb": self.table_gb(),
            "batch": self._batch,
            "last_gate": self._scalar(self._last_gate_t),
            "last_covered": self._scalar(self._last_covered_t),
            "alpha": torch.softmax(self.level_weight, 0).detach().cpu().tolist(),
            "write_gate_bias": float(self.write_gate.bias.detach()),
            "read_gate_bias": float(self.read_gate.bias.detach()),
            # 纯检索口径：覆盖到的样本上，检索 top-1 就是真值的比例
            "retrieval_top1_hit": (self._int_scalar(self._ret_ok_t) / ret_n) if ret_n else float("nan"),
            "retrieval_covered": (self._scalar(self._ret_cover_t) / ret_tot) if ret_tot else float("nan"),
            "n_retrieval": ret_n,
        }

    def extra_repr(self) -> str:
        return (f"levels={self.levels} slots={self.slots} classes={self.num_classes} "
                f"read_true={self.read_true} table(batch={self._batch})={self.table_gb():.3f}GB")
