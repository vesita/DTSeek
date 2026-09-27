"""情绪倾向任务卡：圈出表达情绪的片段，并判断是哪一类情绪。

词典规模决定样本量：199 个情绪词、每类约 60 词，样本量不足时会长尾欠训
（实测词级正确率会从 99% 掉到 43%）。调用方应按词典规模放大 `target_samples`。
"""
from __future__ import annotations

from dtseek.tasks.builtin.sentiment.dataset import LEXICON_BY_CAT, build_sentiment_dataset
from dtseek.tasks.plugin import DEFAULT_SEED, ProbeUnit, TaskClass, TaskSpec, register

SPEC = TaskSpec(
    name="sentiment",
    label="情绪倾向",
    classes=(
        TaskClass("中性"),
        TaskClass("积极", label="积极/喜悦", color="\033[1;92;40m"),
        TaskClass("愤怒", label="愤怒/不满", color="\033[1;91;40m"),
        TaskClass("悲伤", label="悲伤/焦虑", color="\033[1;94;40m"),
    ),
)


class SentimentCard:
    spec = SPEC

    def build_dataset(self, target_samples: int, seed: int = DEFAULT_SEED) -> list[dict]:
        return build_sentiment_dataset(target_samples=target_samples, seed=seed)

    def probe_units(self) -> list[ProbeUnit]:
        """词典里每个词都要考一遍 —— 句级指标看不出「整项没学会」。"""
        return [
            ProbeUnit(key=w, expected_class=cat, words=(w,))
            for cat, words in LEXICON_BY_CAT.items()
            for w in words
        ]

    def sanity_cases(self) -> list[tuple[str, int]]:
        """已知答案对照：这几句在演示里人工验证过，探针必须复现同样结论。"""
        return [
            ("我感觉有点难受", 3),
            ("我有点喜欢这个颜色", 1),
            ("这也太糟心了", 2),
            ("今天心情挺好的", 1),
        ]


CARD = SentimentCard()
register(CARD)
