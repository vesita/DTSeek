"""CPU 冒烟测试：三个参数化臂的前向/反向/梯度/旁路，不占 GPU。"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ngram_memories import build_memory  # noqa: E402

torch.manual_seed(0)
B, L, D, C, V = 4, 24, 16, 8, 8192
input_ids = torch.randint(0, V, (B, L))
mask = torch.ones(B, L, dtype=torch.bool)
h = torch.randn(B, D)
cls_logits = torch.randn(B, C)
starts = torch.randint(0, L, (B,))

for arm in ("lngtab", "ngrammer", "nplm"):
    m = build_memory(arm, D, C, vocab_size=V)
    m.reset(B, "cpu")
    start_logits = torch.randn(B, L)
    attn = m.read_attention(start_logits, mask, starts)    # 教师强制 one-hot
    out = m.read(cls_logits, h, input_ids, attn)
    assert out.shape == cls_logits.shape, (arm, out.shape)
    loss = F.nll_loss(out, torch.randint(0, C, (B,)))
    loss.backward()
    gn = sum(float(p.grad.norm()) for p in m.parameters() if p.grad is not None)
    n_none = [n for n, p in m.named_parameters() if p.grad is None]
    print(f"{arm:9s} params={m.n_params():>8,} grad_total={gn:.4f} no_grad={n_none}")
    assert gn > 0, arm
    assert not n_none, (arm, n_none)
    # 旁路
    m.bypass = True
    out_b = m.read(cls_logits, h, input_ids, attn)
    ref = torch.log_softmax(cls_logits.float(), -1)
    assert torch.allclose(out_b, ref, atol=1e-6), arm
    m.bypass = False
    # 软读回退路径（dense）
    a_soft = m.read_attention(start_logits, mask, None, hard=False)
    out_s = m.read(cls_logits, h, input_ids, a_soft)
    assert out_s.shape == cls_logits.shape
    print(f"          dense-path ok; soft-read differs from onehot: "
          f"{(out_s - out).abs().max().item():.4f}")

print("SMOKE OK")
