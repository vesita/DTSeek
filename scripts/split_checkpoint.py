"""把旧的一体 ckpt（基座 + N 张卡塞在一起）拆成插件化的独立产物。

    输出：
      <out-dir>/base_encoder.pt        任务无关的基座
      <out-dir>/cards/<任务名>.pt       每张任务卡一个文件

拆开之后，加/换任务只需要训那一张卡（`training/train_task_card.py`），不用碰基座。

用法：
    uv run python scripts/split_checkpoint.py --ckpt checkpoints/multitask_frozen_dtseek.pt
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dtseek.tasks.artifacts import BASE_FORMAT, read_base, save_base, save_card  # noqa: E402
from dtseek.tasks.plugin import TaskSpec, all_tasks, check_ckpt_specs  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="拆分一体 ckpt 为基座 + 独立任务卡")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", default=None, help="默认与 ckpt 同目录")
    args = ap.parse_args()

    import torch
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if "task_specs" not in ck or "decoders" not in ck:
        raise SystemExit(f"{args.ckpt} 不是一体 ckpt（缺 task_specs / decoders）")

    out = Path(args.out_dir) if args.out_dir else Path(args.ckpt).parent
    base_path = out / "base_encoder.pt"
    cards_dir = out / "cards"

    save_base(
        base_path,
        _TorchStateDict(ck["doc_encoder"]),
        hidden_dim=ck["hidden_dim"],
        vocab_size=ck.get("vocab_size", 8192),
        encoder_kwargs=ck.get("encoder_kwargs", {}),
        meta={"split_from": str(args.ckpt), "train_args": ck.get("train_args", {})},
    )
    print(f"基座 -> {base_path}")

    registered = all_tasks()
    # 旧快照里缺的推理行为字段，from_snapshot 会填默认值 —— 直接存下去就等于把"缺字段"
    # 写死成"值不同"，挂载时会被判成漂移。所以这里按注册表里的卡解析：
    # 类别名先校验（决定权重含义），推理行为以代码为准。
    check_ckpt_specs(ck["task_specs"], registered)
    for name, sd in ck["decoders"].items():
        card = registered.get(name)
        spec = card.spec if card is not None else TaskSpec.from_snapshot(ck["task_specs"][name])
        path = cards_dir / f"{name}.pt"
        save_card(path, _TorchStateDict(sd), task=name, spec=spec,
                  hidden_dim=ck["hidden_dim"],
                  decoder_kwargs=ck.get("decoder_kwargs", {"num_heads": 4, "num_layers": 2}),
                  base_format=BASE_FORMAT,
                  train_args={"split_from": str(args.ckpt)})
        print(f"  卡 {name:10s} -> {path}  ({spec.num_classes} 类, {path.stat().st_size/1024:.0f} KB)")

    read_base(base_path)          # 立刻回读一遍，确认产物能被自己认出来
    print(f"\n完成。基座 {base_path.stat().st_size/1024/1024:.1f} MB，"
          f"任务卡 {len(ck['decoders'])} 张在 {cards_dir}/")
    return 0


class _TorchStateDict:
    """save_base / save_card 收的是"有 state_dict() 的对象"，这里包一个裸 dict。"""

    def __init__(self, sd):
        self._sd = sd

    def state_dict(self):
        return self._sd


if __name__ == "__main__":
    sys.exit(main())
