import torch
from dtseek import DTSeekConfig, DTSeekModel


def test_dtseek_model_shapes():
    B = 2
    C = 4  # 4 candidate classes
    L = 32  # 32 tokens in document
    D = 128  # hidden dim

    config = DTSeekConfig(hidden_dim=D, num_heads=4, num_decoder_layers=2, use_background_class=True)
    model = DTSeekModel(config)

    class_emb = torch.randn(B, C, D)
    doc_memory = torch.randn(B, L, D)
    doc_mask = torch.ones(B, L, dtype=torch.bool)

    out = model(class_emb, doc_memory, doc_mask=doc_mask)

    assert "logits" in out
    assert "probs" in out
    assert "confidence" in out
    # 4 classes + 1 null background class = 5
    assert out["logits"].shape == (B, C + 1)
    assert out["probs"].shape == (B, C + 1)
    assert out["confidence"].shape == (B, 1)
    assert torch.allclose(out["probs"].sum(dim=-1), torch.ones(B), atol=1e-5)


def test_dtseek_engine_ad_hoc_task():
    # Use CPU lightweight smoke test
    config = DTSeekConfig(hidden_dim=64, num_heads=2, num_decoder_layers=1)
    model = DTSeekModel(config)

    class_emb = torch.randn(1, 3, 64)
    doc_mem = torch.randn(1, 10, 64)

    out = model(class_emb, doc_mem)
    assert out["best_index"].item() in [0, 1, 2, 3]
