"""人称代词任务卡：在句子里圈出人称代词，并判断是第几人称。"""
from __future__ import annotations

from dtseek.tasks.builtin.pronoun.dataset import build_rich_ar_dataset
from dtseek.tasks.plugin import DEFAULT_SEED, ProbeUnit, TaskClass, TaskSpec, register

SPEC = TaskSpec(
    name="pronoun",
    label="人称代词",
    classes=(
        TaskClass("无代词"),
        TaskClass("第一人称", color="\033[1;96;44m"),
        TaskClass("第二人称", color="\033[1;93;41m"),
        TaskClass("第三人称", color="\033[1;97;45m"),
    ),
)

# 代词探针的载体必须自带谓语 —— 裸代词（"我"）没有可判定的语义场，
# 用情绪任务那种空载体测不出东西。
PRONOUN_CARRIERS = (
    "{w}昨天把方案发过来了。",
    "请问{w}对这个接口有什么建议？",
    "这份配置是{w}整理的吧？",
    "{w}刚才说的那个问题已经修复了。",
)

PRONOUN_UNITS = (
    (("我", "我们", "咱们"), 1),
    (("你", "您", "你们"), 2),
    (("他", "她", "他们"), 3),
)


class PronounCard:
    spec = SPEC

    def build_dataset(self, target_samples: int, seed: int = DEFAULT_SEED) -> list[dict]:
        return build_rich_ar_dataset(target_samples=target_samples, seed=seed)

    def probe_units(self) -> list[ProbeUnit]:
        return [
            ProbeUnit(key=w, expected_class=cat, words=(w,), carriers=PRONOUN_CARRIERS)
            for words, cat in PRONOUN_UNITS
            for w in words
        ]


CARD = PronounCard()
register(CARD)
