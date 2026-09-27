"""情绪专卡演示 —— 就是 `example_multitask.py --tasks sentiment` 的固定版本。

类别体系、配色、步数都来自 ckpt 里的 TaskSpec 快照，本文件不再自带 CLASSES 表。

    uv run python examples/example_sentiment.py "开心"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from example_multitask import main  # noqa: E402

if __name__ == "__main__":
    main(default_tasks=["sentiment"])
