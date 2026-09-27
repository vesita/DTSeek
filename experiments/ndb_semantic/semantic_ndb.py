#!/usr/bin/env python
"""semantic-NDB：把 MentionNDB 的键从「字面 n-gram 哈希」换成「语义码本码号」。

**只覆写 `_slots_at`**，其余（表结构、读写顺序、三个门控、读权重必须离散）与
`mention_ndb.MentionNDB` 逐行相同 —— 这样 AB 里「除键的构造外一切相同」是结构上成立的，
不是靠约定。

键 = argmax_k < normalize(W · h_pos) , c_k >，其中
  * h_pos 是**冻结基座**在提及起始位置处的隐状态（`doc_memory[b, pos]`）
  * W 与码本 c_k 都是**冻结**的数据统计量，只从该 seed 的 train split 学出来

`doc_memory` 通过 doc_encoder 的 forward hook 注入（`attach_memory_hook`），
所以本模块对 `runtime.task_loss` / `evaluate_task` / `_rollout` 的调用签名**完全不变**。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "ndb_semantic"))
from mention_ndb_snapshot import MentionNDB  # noqa: E402


class SemanticNDB(MentionNDB):
    """键 = 语义码本码号（VQ），值 = 身份槽 id —— 与字面 NDB 只在键上不同。"""

    def __init__(self, hidden_dim: int, num_classes: int, W: torch.Tensor,
                 centroids: list[torch.Tensor], max_table_gb: float = 0.25,
                 read_true: bool = True):
        ks = [int(c.shape[0]) for c in centroids]
        super().__init__(hidden_dim=hidden_dim, num_classes=num_classes,
                         vocab_size=hidden_dim,       # 键不再用字表；占位以保持签名
                         levels=tuple(range(len(centroids))), slots=ks,
                         max_table_gb=max_table_gb, read_true=read_true)
        self.register_buffer("W", W.float(), persistent=True)
        for i, c in enumerate(centroids):
            self.register_buffer(f"centroid_{i}", c.float(), persistent=True)
        self._centroids = [getattr(self, f"centroid_{i}") for i in range(len(centroids))]
        self._mem: torch.Tensor | None = None
        # 冻结的数据统计量：不进优化器（register_buffer 本来就不是参数）
        self.W.requires_grad_(False)
        for c in self._centroids:
            c.requires_grad_(False)

    def set_memory(self, mem: torch.Tensor) -> None:
        self._mem = mem

    @torch.no_grad()
    def _slots_at(self, input_ids: torch.Tensor, pos: torch.Tensor, li: int) -> torch.Tensor:
        if self._mem is None:
            raise RuntimeError("semantic-NDB 需要 doc_memory：先调用 set_memory / attach_memory_hook")
        B, L, D = self._mem.shape
        pos = pos.clamp(0, L - 1)
        v = self._mem.gather(1, pos.unsqueeze(-1).expand(-1, -1, D))     # [B,P,D]
        z = F.normalize(v @ self.W, dim=-1)
        return (z @ self._centroids[li].T).argmax(-1)                    # [B,P]

    def extra_repr(self) -> str:
        return (f"semantic levels={self.levels} codebook={self.slots} "
                f"classes={self.num_classes} read_true={self.read_true} "
                f"table={self.table_gb():.4f}GB")


def attach_memory_hook(ndb: SemanticNDB, doc_encoder) -> None:
    """把 doc_encoder 每次前向的输出喂给 ndb —— 这样 src 里的调用签名一行都不用改。"""
    def hook(_mod, _inp, out):
        ndb.set_memory(out)
    doc_encoder.register_forward_hook(hook)
