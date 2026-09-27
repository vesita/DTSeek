"""发言归属任务卡：圈出角色标识（"用户"/"模型"/"架构师"…），判断说的是谁。"""
from __future__ import annotations

from dtseek.tasks.builtin.ownership.dataset import SPEAKER_MAP, build_ownership_dataset
from dtseek.tasks.plugin import DEFAULT_SEED, ProbeUnit, TaskClass, TaskSpec, register

SPEC = TaskSpec(
    name="ownership",
    label="发言归属",
    classes=(
        TaskClass("无归属"),
        TaskClass("用户", label="用户/客户", color="\033[1;96;44m"),
        TaskClass("助手", label="助手/系统", color="\033[1;92;40m"),
        TaskClass("第三方", label="第三方/团队", color="\033[1;95;45m"),
    ),
)

# 角色标识裸放（"用户"）不算归属声明，得放进"某某说：…"这种句式里
OWNERSHIP_CARRIERS = (
    "{w}说：这个功能需要再评审一下。",
    "刚才{w}提到接口延迟有点偏高。",
    "请确认一下，这是{w}提交的需求吗？",
    "{w}刚才回复说问题已经解决了。",
)


class OwnershipCard:
    spec = SPEC

    def build_dataset(self, target_samples: int, seed: int = DEFAULT_SEED) -> list[dict]:
        return build_ownership_dataset(target_samples=target_samples, seed=seed)

    def probe_units(self) -> list[ProbeUnit]:
        return [
            ProbeUnit(key=w, expected_class=cat, words=(w,), carriers=OWNERSHIP_CARRIERS)
            for cat, words in SPEAKER_MAP
            for w in words
        ]


CARD = OwnershipCard()
register(CARD)
