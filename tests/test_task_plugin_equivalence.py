"""任务卡插件化重构的等价性验收。

跑 `scripts/dump_equivalence_golden.py` 到临时路径，与仓库里的黄金基线逐字节比对。
三项都在比：数据集内容（dataset_hash）、模块权重（weight_hash）、单步损失（step_loss）。

为什么值得单独一个测试：重构本身的「成功」不能靠读代码断言，只能靠
「重构前后同一份输入产出同一份输出」来证明。任何一项不一致都说明重构动了行为。
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "golden" / "multitask_equivalence.json"

COMPARED = ["task_specs", "dataset_hash", "decoder_weight_hash", "step_loss", "encoder_weight_hash"]


@pytest.fixture(scope="module")
def regenerated(tmp_path_factory) -> dict:
    out = tmp_path_factory.mktemp("golden") / "regen.json"
    proc = subprocess.run(
        [sys.executable, "scripts/dump_equivalence_golden.py", "--out", str(out)],
        cwd=ROOT, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, f"黄金值生成失败：\n{proc.stdout}\n{proc.stderr}"
    return json.loads(out.read_text(encoding="utf-8"))


def test_golden_matches_recorded_baseline(regenerated):
    baseline = json.loads(GOLDEN.read_text(encoding="utf-8"))
    diffs = [k for k in COMPARED if baseline.get(k) != regenerated.get(k)]
    if diffs:
        detail = "\n".join(
            f"  {k}:\n    基线={json.dumps(baseline.get(k), ensure_ascii=False)[:200]}"
            f"\n    现在={json.dumps(regenerated.get(k), ensure_ascii=False)[:200]}"
            for k in diffs)
        pytest.fail("重构改动了行为，以下观测量不一致：\n" + detail)
