r"""情绪任务词级探针 —— 就是 `scripts/eval_probe.py --task sentiment`。

保留这个文件名只是为了让旧命令还能跑；探针逻辑本身在 `dtseek/tasks/probe.py`，
任务无关，情绪卡的待测词由 `SentimentCard.probe_units()` 声明。

    uv run python scripts/eval_emotion_words.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_probe import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["--task", "sentiment", *sys.argv[1:]]))
