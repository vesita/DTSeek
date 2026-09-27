"""Mention-NDB 的单测：键对齐、写 / 读、门控、指针不受影响、显存硬上限。

每个测量函数都配一个**已知答案的对照**：键的对齐用「在 p 位置写、在 p 位置读得到、
在别的不同字位置读不到」钉死（历史上 n-gram 查表最常错的就是 ±1 对齐）；
指针不受影响用「ndb 读路径 backward 后 start_ptr/end_ptr 无梯度」钉死。
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from dtseek.decoder.mention_ndb import MentionNDB
from dtseek.decoder.robust_ar_model import RobustARSliceDecoder

VOCAB = 64
C = 8          # 0 = 背景 + 7 个身份槽


def _ndb(**kw) -> MentionNDB:
    base = dict(hidden_dim=8, num_classes=C, vocab_size=VOCAB,
                levels=(1, 2), slots=(VOCAB, 32), max_table_gb=0.05)
    base.update(kw)
    n = MentionNDB(**base)
    # 门控权重清零 → 门控只由 bias 决定，测试才有确定答案
    with torch.no_grad():
        n.write_gate.weight.zero_()
        n.read_gate.weight.zero_()
    return n


def _open_gate(n: MentionNDB, bias: float = 20.0) -> None:
    """把读门控强制全开（bias 走 sigmoid，20 ≈ 1.0）。"""
    with torch.no_grad():
        n.read_gate.bias.fill_(bias)


def _doc(B: int = 2, L: int = 16) -> torch.Tensor:
    """每条样本一个逐位置互不相同的字，避免哈希碰撞干扰对齐判据。"""
    row = torch.arange(L) % VOCAB
    return row.unsqueeze(0).expand(B, L).contiguous()


def _onehot(pos: int, L: int = 16, B: int = 2, mask: torch.Tensor | None = None) -> torch.Tensor:
    a = F.one_hot(torch.full((B,), pos), L).float()
    if mask is not None:
        a = a * mask.float()
    return a / a.sum(-1, keepdim=True).clamp_min(1e-9)


def test_hard_cap_raises_before_allocating():
    """表超上限必须抛错，而不是跑到一半 OOM。"""
    n = _ndb(slots=(1 << 14, 1 << 14), max_table_gb=0.001)
    with pytest.raises(ValueError, match="超过上限"):
        n.reset(batch_size=64, device="cpu")


def test_write_then_read_same_position_hits():
    """★ 对齐：在 p 写的绑定，只有在 p 处读才命中（±1 都不行）。"""
    n = _ndb()
    doc = _doc()
    h = torch.zeros(2, 8)
    n.reset(2, "cpu")
    with n.write_enabled():
        n.write(h, doc, torch.tensor([3, 3]), torch.tensor([5, 6]), torch.ones(2))
    assert n.n_written == 2
    _open_gate(n)          # 门控全开
    cls = torch.zeros(2, C)
    cls[:, 1] = 5.0                       # 分类头自己的答案完全不是检索答案
    logp = n.read(cls, h, doc, _onehot(3))
    assert logp.argmax(-1).tolist() == [5, 6], "在写入位置读不出写进去的 id = 键错位"

    # 换一个「字面不同」的位置读：槽为空 → 门控归零 → 退回分类头自己的预测
    n.reset(2, "cpu")
    with n.write_enabled():
        n.write(h, doc, torch.tensor([3, 3]), torch.tensor([5, 6]), torch.ones(2))
    logp2 = n.read(cls, h, doc, _onehot(9))
    assert logp2.argmax(-1).tolist() == [1, 1], "未写入的位置不该被检索覆盖"


def test_level2_key_distinguishes_shared_first_char():
    """一级键（首个字）会撞车，二级键（前两个字）不会 —— 这正是多级混合的理由。"""
    n = _ndb()
    doc = torch.tensor([[7, 11, 0], [7, 12, 0]])       # 「小张」vs「小李」：首字相同
    pos = torch.tensor([[0], [0]])
    s1 = n._slots_at(doc, pos, 0).flatten()
    s2 = n._slots_at(doc, pos, 1).flatten()
    assert s1[0] == s1[1], "一级键按定义应碰撞"
    assert s2[0] != s2[1], "二级键必须区分共享首字的不同人"


def test_gate_zeroed_where_uncovered_and_logprob_is_valid_logits():
    """空表 → p_new == p_model：log-prob 口径与原来的 cross_entropy 完全等价。"""
    n = _ndb()
    doc = _doc()
    h = torch.zeros(2, 8)
    n.reset(2, "cpu")
    _open_gate(n)          # 就算门控想开，槽为空也必须归零
    cls = torch.randn(2, C)
    logp = n.read(cls, h, doc, _onehot(3))
    assert torch.allclose(logp, F.log_softmax(cls, -1), atol=1e-6)
    y = torch.tensor([1, 6])
    assert torch.allclose(F.cross_entropy(logp, y), F.cross_entropy(cls, y), atol=1e-6)


def test_write_requires_explicit_context_and_skips_background():
    """写必须显式开启；label=0（背景）与无效步都不写。"""
    n = _ndb()
    doc = _doc()
    h = torch.zeros(2, 8)
    n.reset(2, "cpu")
    assert n.write(h, doc, torch.tensor([3, 3]), torch.tensor([5, 6]), torch.ones(2)) is None
    assert n.n_written == 0
    with n.write_enabled():
        n.write(h, doc, torch.tensor([3, 3]), torch.tensor([0, 6]), torch.tensor([1.0, 0.0]))
    assert n.n_written == 0, "背景 / 无效步不该进表"


def test_pointer_logits_untouched_and_no_grad_to_pointers():
    """★ 硬约束：读路径不改 start/end 指针的数值，也不给它任何梯度。"""
    torch.manual_seed(0)
    dec = RobustARSliceDecoder(hidden_dim=16, num_classes=C, num_heads=2, num_layers=1,
                               dim_feedforward=32)
    dec.train()
    doc = _doc()
    mem = torch.randn(2, 16, 16)
    mask = torch.ones(2, 16, dtype=torch.bool)
    q = dec.bos_query.expand(2, 1, -1)
    out = dec.forward_step(q, mem, doc_mask=mask)
    start_before = out["start_logits"].clone()
    end_before = out["end_logits"].clone()

    n = _ndb(hidden_dim=16)
    n.reset(2, "cpu")
    _open_gate(n)
    loss = n.read(out["cls_logits"], out["last_hidden"].squeeze(1), doc,
                  n.read_attention(out["start_logits"], mask, torch.tensor([3, 3]))).sum()
    loss.backward()

    assert torch.equal(out["start_logits"], start_before)
    assert torch.equal(out["end_logits"], end_before)
    assert dec.start_ptr.weight.grad is None, "读注意力没 detach，指针被 NDB 带上了梯度"
    assert dec.end_ptr.weight.grad is None
    assert dec.cls_head.weight.grad is not None, "分类头本来就该收到梯度"


def test_read_attention_read_true_is_one_hot_at_true_start():
    n = _ndb()
    n.eval()
    start_logits = torch.randn(2, 16)
    mask = torch.ones(2, 16, dtype=torch.bool)
    a = n.read_attention(start_logits, mask, torch.tensor([3, 11]))
    assert torch.equal(a.argmax(-1), torch.tensor([3, 11]))
    assert torch.allclose(a.sum(-1), torch.ones(2))
    a_pred = n.read_attention(start_logits, mask, None)
    assert torch.allclose(a_pred.argmax(-1), start_logits.argmax(-1))


def test_read_before_reset_raises():
    n = _ndb()
    with pytest.raises(RuntimeError, match="reset"):
        n.read(torch.zeros(1, C), torch.zeros(1, 8), _doc(1), _onehot(3, B=1))


def test_engine_rebuilds_ndb_from_card_artifact():
    """★ 推理端必须把 NDB 一起挂上：只挂 decoder 会静默丢掉记忆（指标掉回基线且不报错）。"""
    from dtseek.tasks.engine import load_card_ndb

    assert load_card_ndb({}, "cpu") is None
    assert load_card_ndb({"extra": {}}, "cpu") is None

    n = _ndb()
    kwargs = dict(hidden_dim=8, num_classes=C, vocab_size=VOCAB, levels=(1, 2),
                  slots=(VOCAB, 32), max_table_gb=0.05)
    ck = {"extra": {"ndb": {"kwargs": kwargs, "state_dict": n.state_dict()}}}
    got = load_card_ndb(ck, "cpu")
    assert isinstance(got, MentionNDB)
    assert not got.training, "推理端载入的记忆必须是 eval 模式"
    assert torch.allclose(got.read_gate.bias, n.read_gate.bias)


def test_onehot_fast_path_bitwise_matches_dense_path():
    """★ 单点 gather 路径与稠密路径必须**逐位**相同（含整行被 mask 掉、真值起点、软读三支）。"""
    n = _ndb()
    doc = _doc()
    h = torch.randn(2, 8)
    cls = torch.randn(2, C)
    n.reset(2, "cpu")
    with n.write_enabled():
        n.write(h, doc, torch.tensor([3, 5]), torch.tensor([5, 6]), torch.ones(2))
    _open_gate(n)
    start_logits = torch.randn(2, 16)
    mask = torch.ones(2, 16, dtype=torch.bool)
    mask[1, int(start_logits[1].argmax())] = False       # 第 1 行整行被 mask 掉
    a = n.read_attention(start_logits, mask)
    assert n._fast_attn is a, "one-hot 注意力必须登记成可信单点路径"
    fast = n.read(cls, h, doc, a)                        # 同一个对象 → 单点路径
    dense = n.read(cls, h, doc, a.clone())               # 换个对象 → 稠密路径
    assert torch.equal(fast, dense), f"max|Δ|={(fast - dense).abs().max():.3e}"
    # 教师强制（真值起点）分支同样逐位一致
    a2 = n.read_attention(start_logits, mask, torch.tensor([3, 5]))
    assert torch.equal(n.read(cls, h, doc, a2), n.read(cls, h, doc, a2.clone()))
    # 软读诊断不登记单点元数据，必须回退稠密路径且能跑
    a3 = n.read_attention(start_logits, mask, hard=False)
    assert n._fast_attn is None, "软读不能走单点路径（会把新人物判成已覆盖）"
    assert torch.equal(n.read(cls, h, doc, a3), n.read(cls, h, doc, a3))


def test_merged_table_places_counts_at_documented_flat_slot():
    """两级共用一张 [B, Σslots, C] 表：级 li 的槽位 s 落在 flat = offset_li + s。"""
    n = _ndb()
    doc = _doc()
    h = torch.zeros(2, 8)
    starts = torch.tensor([3, 5])
    n.reset(2, "cpu")
    with n.write_enabled():
        n.write(h, doc, starts, torch.tensor([5, 6]), torch.ones(2))
    with torch.no_grad():
        w = torch.sigmoid(n.write_gate(h)).squeeze(-1)
    ar = torch.arange(2)
    for li in range(len(n.levels)):
        flat = n._slots_at(doc, starts.view(2, 1), li).view(2) + n._offsets[li]
        assert torch.equal(n._tab_tot[ar, flat], w)
        assert n._counts[li][ar, flat - n._offsets[li]].argmax(-1).tolist() == [5, 6]
        assert n._counts[li].shape == (2, n.slots[li], C)
        assert n._totals[li].shape == (2, n.slots[li])


def test_diagnostics_are_lazy_but_keep_exact_values():
    """诊断量只在读取时物化（不再每步 .item()），但值必须与旧实现完全一样。"""
    n = _ndb()
    doc = _doc()
    h = torch.zeros(2, 8)
    n.reset(2, "cpu")
    _open_gate(n)
    assert n.n_written == 0
    with n.write_enabled():
        # 第一笔：label=0（背景）与 valid=0 都不写
        n.write(h, doc, torch.tensor([3, 3]), torch.tensor([0, 6]), torch.tensor([1.0, 0.0]))
        assert n.n_written == 0
        n.write(h, doc, torch.tensor([3, 3]), torch.tensor([5, 6]), torch.ones(2))
    assert n.n_written == 2
    st = n.stats()
    assert st["n_written"] == 2 and st["batch"] == 2
    n.reset_stats()
    assert n.n_written == 0
    # 写入关闭时仍然是 None（旧行为里唯一的返回值判据）
    assert n.write(h, doc, torch.tensor([3, 3]), torch.tensor([5, 6]), torch.ones(2)) is None


def test_out_of_range_start_is_clamped_like_old_keys():
    """`engine.py` 会传 `min(len(segment_text)-1, ...)`，可能越出 padded 窗口右端。

    旧 `_keys` 用 `pos.clamp(0, last)` 把它夹到最后一个位置；预计算键路径必须一致，
    既不能崩，也不能落到别的槽位上。
    """
    n = _ndb()
    doc = _doc()
    h = torch.zeros(2, 8)
    n.reset(2, "cpu")
    with n.write_enabled():
        n.write(h, doc, torch.tensor([3, 999]), torch.tensor([5, 6]), torch.ones(2))
    assert n.n_written == 2
    last = doc.shape[1] - 1
    ar = torch.arange(2)
    for li in range(len(n.levels)):
        flat = n._slots_at(doc, torch.tensor([[3], [last]]), li).view(2) + n._offsets[li]
        assert n._tab_tot[ar, flat].min() > 0, "越界起点必须夹到最后一个位置"
