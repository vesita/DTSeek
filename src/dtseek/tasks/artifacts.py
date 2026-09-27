"""任务卡产物格式：基座与任务卡**分开存**。

插件化的意思是"换任务类型立刻换"，所以两者必须是独立文件：

    base_encoder.pt      任务无关的文本编码器（训一次、冻结、所有卡共用）
    cards/<任务名>.pt      单张任务卡（解码头 + 它的 TaskSpec 快照 + 训练信息）

曾经的格式是"基座 + N 张卡塞进同一个 ckpt"，那样加一张新卡必须重训整个基座 ——
这正是要修掉的东西。`scripts/split_checkpoint.py` 可以把旧的一体 ckpt 拆成新格式。

放在一处定义的意义：训练侧（`train_task_card.py`）、推理侧（`engine.py`）、
迁移脚本三边共用同一套读写与校验，不会各写一份后悄悄分叉。
"""
from __future__ import annotations

from pathlib import Path

import torch

BASE_FORMAT = "dtseek.base_encoder.v1"
CARD_FORMAT = "dtseek.task_card.v1"


class ArtifactError(ValueError):
    """产物格式不对 / 版本不符 / 卡与基座不兼容。"""


# ---- 基座 -----------------------------------------------------------------

def save_base(path: str | Path, doc_encoder, *, hidden_dim: int, vocab_size: int,
              encoder_kwargs: dict, meta: dict | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": BASE_FORMAT,
        "doc_encoder": doc_encoder.state_dict(),
        "hidden_dim": hidden_dim,
        "vocab_size": vocab_size,
        "encoder_kwargs": encoder_kwargs,
        "meta": meta or {},
    }, path)


def read_base(path: str | Path) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    fmt = ck.get("format")
    if fmt != BASE_FORMAT:
        raise ArtifactError(
            f"{path} 不是基座产物（format={fmt!r}，期望 {BASE_FORMAT!r}）。"
            " 如果这是旧的一体 ckpt，先用 scripts/split_checkpoint.py 拆开。")
    return ck


def load_base_encoder(path: str | Path, device="cpu"):
    """按产物里记的 kwargs 重建编码器并载入权重，返回 (encoder, ckpt_dict)。"""
    from dtseek.encoder.nano_doc_encoder import NanoDocEncoder

    ck = read_base(path)
    enc = NanoDocEncoder(
        vocab_size=ck["vocab_size"],
        hidden_dim=ck["hidden_dim"],
        dropout=0.0,
        **ck["encoder_kwargs"],
    ).to(device)
    enc.load_state_dict(ck["doc_encoder"])
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)          # 基座永远是冻结的：这是插件化的前提
    return enc, ck


# ---- 任务卡 ---------------------------------------------------------------

def save_card(path: str | Path, decoder, *, task: str, spec, hidden_dim: int,
              decoder_kwargs: dict, base_format: str = BASE_FORMAT,
              train_args: dict | None = None, extra: dict | None = None) -> None:
    """存一张任务卡。

    `extra` 是**可选的外挂模块附加信息**（如人物卡的 Mention-NDB 门控权重与配置）。
    默认空 dict：不传时产物与旧版逐字节等价，旧读端也完全不受影响。
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": CARD_FORMAT,
        "task": task,
        "spec": spec.to_snapshot(),
        "decoder": decoder.state_dict(),
        "hidden_dim": hidden_dim,
        "decoder_kwargs": decoder_kwargs,
        "base_format": base_format,
        "train_args": train_args or {},
        "extra": extra or {},
    }, path)


def read_card(path: str | Path) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    fmt = ck.get("format")
    if fmt != CARD_FORMAT:
        raise ArtifactError(
            f"{path} 不是任务卡产物（format={fmt!r}，期望 {CARD_FORMAT!r}）。")
    return ck


def build_card_decoder(ck: dict, device="cpu"):
    from dtseek.decoder.robust_ar_model import RobustARSliceDecoder

    from dtseek.tasks.plugin import TaskSpec

    spec = TaskSpec.from_snapshot(ck["spec"])
    dec = RobustARSliceDecoder(hidden_dim=ck["hidden_dim"],
                               num_classes=spec.num_classes,
                               **ck["decoder_kwargs"]).to(device)
    dec.load_state_dict(ck["decoder"])
    dec.eval()
    return dec, spec


def check_card_base_compat(card_ck: dict, base_ck: dict, card_path: str = "") -> None:
    """挂载前校验：卡的隐藏维与基座对得上，且格式版本匹配。"""
    problems = []
    if card_ck["hidden_dim"] != base_ck["hidden_dim"]:
        problems.append(f"hidden_dim 不一致：卡={card_ck['hidden_dim']} vs 基座={base_ck['hidden_dim']}")
    if card_ck.get("base_format") != base_ck.get("format"):
        problems.append(f"基座格式不一致：卡要求={card_ck.get('base_format')!r} vs 基座={base_ck.get('format')!r}")
    if problems:
        raise ArtifactError(
            f"任务卡 {card_path or card_ck.get('task')} 与基座不兼容：" + "；".join(problems))
