"""记录多任务栈的等价性黄金值。

把「数据集内容 + 基座/任务卡权重 + 单步损失」三样落盘；任务卡插件化重构
**不允许改动这三样**，所以这个脚本重构前后跑出来必须逐字节一致。

为什么不是「跑完训练比指标」：
训练本身带噪，比不出重构是否走样。这里改成三个确定性更强的观测量——
  1. dataset_hash : 三个数据集构建器都是固定 seed 的确定性函数，内容必须逐字节一致
  2. weight_hash  : 每个模块用**独立** seed 构造 ⇒ 与模块构造顺序无关，只取决于模块定义
  3. step_loss    : 关掉 dropout 的纯函数，只取决于权重与数据

用法：uv run python scripts/dump_equivalence_golden.py
     然后用 `git diff tests/golden/` 确认没有变化。
"""
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nano_char_tokenizer import NanoCharTokenizer  # noqa: E402
from dtseek.encoder.nano_doc_encoder import NanoDocEncoder  # noqa: E402
from dtseek.decoder.robust_ar_model import RobustARSliceDecoder  # noqa: E402
from dtseek.tasks.builtin.pronoun.dataset import build_rich_ar_dataset  # noqa: E402
from dtseek.tasks.builtin.sentiment.dataset import build_sentiment_dataset  # noqa: E402
from dtseek.tasks.builtin.ownership.dataset import build_ownership_dataset  # noqa: E402
from dtseek.tasks.plugin import resolve_tasks  # noqa: E402
from dtseek.tasks.runtime import GenericTaskDataset, task_loss  # noqa: E402

OUT = ROOT / "tests" / "golden" / "multitask_equivalence.json"

# 小配置只为跑得快；确定性才是目的，与生产配置无关
DATASET_BUILDERS = {
    "pronoun": lambda: build_rich_ar_dataset(target_samples=200),
    "sentiment": lambda: build_sentiment_dataset(
        target_samples=200, per_word_floor=1, per_word_cap=2),
    "ownership": lambda: build_ownership_dataset(target_samples=200),
}
HIDDEN = 64


def hash_dataset(data: list[dict]) -> str:
    """按 (文本, 有序 span 的 类别/起止) 逐字节哈希，忽略 span 里的附带字段。"""
    h = hashlib.sha256()
    for item in data:
        h.update(item["text"].encode("utf-8"))
        h.update(b"\x1f")
        for s in sorted(item["spans"], key=lambda x: (x["start"], x["end"], x["label"])):
            h.update(f"{s['label']}:{s['start']}:{s['end']}".encode())
            h.update(b",")
        h.update(b"\n")
    return h.hexdigest()


def hash_state_dict(sd: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(str(tuple(sd[k].shape)).encode())
        h.update(sd[k].detach().to(torch.float64).numpy().tobytes())
    return h.hexdigest()


def build_stack(cards, seed_base: int):
    """每个模块单独播种 —— 这样等价性只取决于模块定义，与构造顺序无关。"""
    torch.manual_seed(seed_base)
    encoder = NanoDocEncoder(vocab_size=NanoCharTokenizer().vocab_size,
                             hidden_dim=HIDDEN, num_layers=1, num_heads=4,
                             max_len=128, dropout=0.0)
    decoders = {}
    for i, name in enumerate(sorted(cards)):  # 排序取种子：与注册顺序解耦，重构才不会误报
        torch.manual_seed(seed_base + 100 + i)
        decoders[name] = RobustARSliceDecoder(
            hidden_dim=HIDDEN, num_classes=cards[name].spec.num_classes,
            num_heads=4, num_layers=1)
    return encoder, decoders


def step_loss(encoder, decoder, dataset, tokenizer, spec) -> float:
    """eval 模式下的一次前向 —— dropout 关闭，输出是权重与数据的纯函数。"""
    encoder.eval()
    decoder.eval()
    ds = GenericTaskDataset(dataset[:8], tokenizer, spec)
    batch = {k: torch.stack([ds[i][k] for i in range(len(ds))]) for k in ds[0]}
    with torch.no_grad():
        mem = encoder(batch["input_ids"], attention_mask=batch["attention_mask"])
        loss = task_loss(decoder, mem, batch["attention_mask"], batch, spec, torch.device("cpu"))
    return float(loss)


def main(out_path: Path | None = None):
    cards = resolve_tasks(["pronoun", "sentiment", "ownership"])
    tokenizer = NanoCharTokenizer()
    encoder, decoders = build_stack(cards, 1234)

    golden = {
        "note": "任务卡插件化重构的等价性基线：不得改动下面任何一项",
        "hidden_dim": HIDDEN,
        "max_steps": 4,
        "task_specs": {n: list(c.spec.class_names) for n, c in cards.items()},
        "encoder_weight_hash": hash_state_dict(encoder.state_dict()),
        "dataset_hash": {},
        "decoder_weight_hash": {},
        "step_loss": {},
    }

    for name, build in DATASET_BUILDERS.items():
        print(f"构建 {name} 数据集 ...")
        data = build()
        golden["dataset_hash"][name] = hash_dataset(data)
        print(f"  {len(data)} 条, hash={golden['dataset_hash'][name][:16]}")

    for name in sorted(cards):
        card = cards[name]
        decoder = decoders[name]
        golden["decoder_weight_hash"][name] = hash_state_dict(decoder.state_dict())
        golden["step_loss"][name] = step_loss(encoder, decoder, DATASET_BUILDERS[name](),
                                              tokenizer, card.spec)
        print(f"  [{name:9s}] weights={golden['decoder_weight_hash'][name][:16]} "
              f"loss={golden['step_loss'][name]:.8f}")

    out = out_path or OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(golden, f, ensure_ascii=False, indent=2)
    print(f"\n黄金值已写入 {out}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="记录多任务栈的等价性黄金值")
    ap.add_argument("--out", type=Path, default=None, help="输出路径；默认覆盖 tests/golden/")
    main(ap.parse_args().out)
