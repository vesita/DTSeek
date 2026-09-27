"""成语识别任务卡：在句子里圈出所有成语。

类别体系只有两类（背景 / 成语），是**逐切片发射**任务 —— 一句话里 0 到 N 个，
模型自己决定吐几个。成对任务（`relation`）验证的是「两个切片构成一对」，
这张卡验证的是「切片数完全可变」，两者合起来覆盖了自回归切片发射的主要形态。

正面语料是词典例句（文学语体），背景取自词典出处引文与释义（同样偏书面）——
语体对齐是刻意的：若正负样本一个文学一个口语，模型会学成语体差异而不是成语本身。

已知取舍：白名单来自开源词典，其中混入了少量现代词组（AABB 重叠式、「～主义」、
口语短语）。它们**在训练与验收里是同一套**，所以任务自洽；但严格意义上
「成语」的边界比白名单更窄。要收紧应加规则而不是逐条删。
"""
from __future__ import annotations

from dtseek.tasks.builtin.idiom.dataset import build_idiom_dataset
from dtseek.tasks.builtin.idiom.lexicon import IDIOMS
from dtseek.tasks.plugin import DEFAULT_SEED, ProbeUnit, TaskClass, TaskSpec, register

SPEC = TaskSpec(
    name="idiom",
    label="成语识别",
    classes=(
        TaskClass("非成语"),
        TaskClass("成语", color="\033[1;95;45m"),
    ),
    max_steps=4,
)

# 与训练用的 SYNTH_FRAMES 刻意不重合 —— 两边都用 "{w}。" 就等于自己考自己。
PROBE_FRAMES = (
    "古书里常有{w}这样的说法。",
    "你能说出{w}的出处吗？",
    "说到这个道理，他想起了{w}。",
    "第一页上印着{w}。",
    "同学们齐声念道：{w}。",
)


class IdiomCard:
    spec = SPEC

    def build_dataset(self, target_samples: int, seed: int = DEFAULT_SEED) -> list[dict]:
        return build_idiom_dataset(target_samples=target_samples, seed=seed)

    def probe_units(self) -> list[ProbeUnit]:
        """2214 个成语逐个考；载体是不含语义线索的通用句式。"""
        return [
            ProbeUnit(key=w, expected_class=1, words=(w,), carriers=PROBE_FRAMES)
            for w in sorted(IDIOMS)
        ]


CARD = IdiomCard()
register(CARD)
