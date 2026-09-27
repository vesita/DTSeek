"""语义关系任务卡：在一句话里圈出一对词，并判断这对是近义词还是反义词。

**成对发射**（`emission="pair"`）：连续两个切片构成一对，左先右后，两个切片共享
同一个类别。这样成对关系由发射顺序隐式表达，解码器不用改 —— 比加 4 指针配对头
省事得多，代价是必须验证模型不会吐出奇数个切片（见 `evaluate_task` 的 `pair_odd`）。

类别体系：
  0 无关系（背景：句子里没有成对的词）
  1 近义词对
  2 反义词对

探针用的是**和训练不重合的帧**：如果模型只学会了"这个句式 → 这个类别"而没学会
两个词本身的关系，换帧之后准确率会塌 —— 这正是探针要暴露的事。
"""
from __future__ import annotations

from dtseek.tasks.builtin.relation.lexicon import ANTONYM_PAIRS, SYNONYM_PAIRS
from dtseek.tasks.builtin.relation.dataset import build_relation_dataset
from dtseek.tasks.plugin import DEFAULT_SEED, ProbeUnit, TaskClass, TaskSpec, register

SPEC = TaskSpec(
    name="relation",
    label="词义关系",
    classes=(
        TaskClass("无关系"),
        TaskClass("近义词", color="\033[1;92;40m"),
        TaskClass("反义词", color="\033[1;91;40m"),
    ),
    max_steps=4,
    emission="pair",
)

# 与训练帧（relation_dataset.NEUTRAL_FRAMES / LEAN_FRAMES）**刻意不重合**，
# 用来测「换一个句式还认不认得出这对词的关系」。
PROBE_FRAMES = (
    "{w}、{w2}，这两个记下来。",
    "请把{w}和{w2}各抄两遍。",
    "他读了一遍{w}，又读了一遍{w2}。",
    "黑板上并排写着{w}和{w2}。",
)


class RelationCard:
    spec = SPEC

    def build_dataset(self, target_samples: int, seed: int = DEFAULT_SEED) -> list[dict]:
        return build_relation_dataset(target_samples=target_samples, seed=seed)

    def probe_units(self) -> list[ProbeUnit]:
        """每一对词都要考一遍，且用的是没见过的句式。"""
        return [
            ProbeUnit(key=f"{a}|{b}", expected_class=cat, words=(a, b), carriers=PROBE_FRAMES)
            for cat, pairs in ((1, SYNONYM_PAIRS), (2, ANTONYM_PAIRS))
            for a, b in pairs
        ]


CARD = RelationCard()
register(CARD)
