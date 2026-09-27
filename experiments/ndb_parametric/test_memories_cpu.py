#!/usr/bin/env python
"""CPU 自检：三个记忆臂的鸭子接口形状 / 梯度非零 / 旁路 / 全局 RNG 不被消耗。

不依赖 GPU、不依赖训练循环，用来在跑 2 分钟一次的 AB 之前拦住形状类 bug。

用法：uv run python experiments/ndb_parametric/test_memories_cpu.py
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import memories as M  # noqa: E402

D, C, L, B, S = 16, 5, 9, 4, 3


def rng_digest():
    return hashlib.md5(torch.get_rng_state().numpy().tobytes()).hexdigest()


def build(arm):
    if arm == "slots16":
        return M.SlotsMemory(D, C, n_slots=16, seed=7)
    if arm == "slots32":
        return M.SlotsMemory(D, C, n_slots=32, seed=7)
    if arm == "diffwrite":
        return M.DiffWriteMemory(D, C, seed=7)
    if arm == "fastweight":
        return M.FastWeightMemory(D, C, seed=7, key_dim=8, value_from_cls=False)
    if arm == "fastweight_cls":
        return M.FastWeightMemory(D, C, seed=7, key_dim=8, value_from_cls=True)
    raise ValueError(arm)


def run(arm):
    torch.manual_seed(0)
    h0 = rng_digest()
    mem = build(arm)
    assert rng_digest() == h0, f"{arm}: 构造消耗了全局 RNG"
    mem.reset(B, "cpu")
    input_ids = torch.randint(0, 100, (B, L))
    mask = torch.ones(B, L)
    starts = torch.tensor([1, 2, 3, 4])
    labels = torch.tensor([1, 2, 3, 0])
    valid = torch.ones(B)
    torch.manual_seed(123)
    h = torch.randn(B, D)

    for step in range(S):
        cls = torch.randn(B, C)
        attn = mem.read_attention(torch.randn(B, L), mask, starts)
        assert attn.shape == (B, L), f"{arm}: read_attention 形状 {attn.shape}"
        out = mem.read(cls, h, input_ids, attn)
        assert out.shape == cls.shape, f"{arm}: read 形状 {out.shape}"
        with mem.write_enabled():
            mem.write(h, input_ids, starts, labels, valid)

    loss = mem.read(torch.randn(B, C), h, input_ids, attn).sum()
    loss.backward()
    norms = {}
    for n, p in mem.named_parameters():
        norms[n] = None if p.grad is None else float(p.grad.norm())
    # 已知无梯度参数：fastweight_cls 的值来自 onehot(cls)，投影 W_v 用不上。
    # 保留它（而不是删掉）是为了让代码与 ab_parametric 已产出的结果逐位对应；
    # 代价是它在参数统计里占 1024 个（见 README「已知瑕疵」）。
    dead_ok = {"W_v"} if getattr(mem, "value_from_cls", False) else set()
    bad = [n for n, v in norms.items() if v is None and n not in dead_ok]
    assert not bad, f"{arm}: 非预期无梯度参数 {bad}（全部 {norms}）"
    total = sum(v for v in norms.values() if v is not None)
    assert total > 0, f"{arm}: 全部梯度为 0（空模块）"

    # 旁路：read 必须原样返回 cls_logits
    cls = torch.randn(B, C)
    mem.bypass = True
    assert mem.read(cls, h, input_ids, attn) is cls, f"{arm}: bypass 没生效"
    mem.bypass = False
    n_written = mem.stats()["n_written"]
    print(f"  OK {arm:16s} params={sum(p.numel() for p in mem.parameters()):5d} "
          f"grad_total={total:.4f} n_written={n_written} stats={sorted(mem.stats())}")
    return norms


if __name__ == "__main__":
    for a in ("slots16", "slots32", "diffwrite", "fastweight", "fastweight_cls"):
        run(a)
    print("CPU 自检全部通过")
