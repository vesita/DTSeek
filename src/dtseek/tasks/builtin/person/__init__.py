"""人物追踪任务卡：在一段文本里标出**所有**人物提及，并给每个提及一个人物 id。

**为什么不预设人物**：旧归属卡把类别写死成「用户 / 助手 / 第三方」，模型学到的是
「哪些字面像角色标记」，而不是「谁和谁是同一个人」。这张卡改成**匿名身份槽**：

  - 类别只有 `无人物` + `人物1..人物7`，槽位没有任何语义；
  - id 按**段落内出场顺序**分配，第一个出场的人物 = 1，第二个 = 2，依此类推；
  - 同一人物的后续提及必须复用同一个 id。

于是这张卡真正要做的是**共指**，而不只是分类。用户报的那个失败例子正好落在这里：

      他说：“我不认识你”
      -> 他(人物1) 我(人物1) 你(人物2)      # 他 与 我 是同一个人

代词卡只能说 第三人称 / 第一人称 / 第二人称（句法层面），说不了「他和我是一个人」。

**验收测的不是「id 猜得准不准」**（留空槽位的 id 本来就可以靠出场顺序推），
而是「同一个人物的多次提及有没有落到同一个 id 上」——见 `TaskSpec.identity_labels`：

  - `first_mention_acc`  : 首次提及（能靠顺序推）
  - `repeat_mention_acc` : **重复提及**（这才是「id 有没有对上」）
  - `cluster_f1`         : 聚类一致性，对 id 重命名不敏感

窗口取 120 字（基座支持到 128），比别的任务大一倍，用来装多轮上下文。
"""
from __future__ import annotations

from dtseek.tasks.builtin.person.dataset import build_person_dataset
from dtseek.tasks.plugin import DEFAULT_SEED, ProbeUnit, TaskClass, TaskSpec, register

#: 身份槽数量（不含背景）。段落里人物超过这个数就不可表达，数据集侧负责丢弃。
MAX_PERSON_SLOTS = 7

_SLOT_COLORS = (
    "\033[1;96;44m",   # 亮青
    "\033[1;93;41m",   # 亮黄
    "\033[1;97;45m",   # 亮白
    "\033[1;92;40m",   # 亮绿
    "\033[1;91;40m",   # 亮红
    "\033[1;94;40m",   # 亮蓝
    "\033[1;95;40m",   # 亮紫
)

SPEC = TaskSpec(
    name="person",
    label="人物追踪",
    classes=(TaskClass("无人物"),)
    + tuple(TaskClass(f"人物{i}", color=_SLOT_COLORS[i - 1]) for i in range(1, MAX_PERSON_SLOTS + 1)),
    max_steps=16,
    max_len=120,
    # 样本里的人物提及必须标全：被 max_steps 截掉一半 = 漏标，不能静默发生
    annotate_all=True,
    # 跨句共指：窗口内多句必须一起解码，否则 id 每句从 1 重来
    segment_policy="window",
    # 类别是匿名身份槽：要测的是「同一人物的多次提及是否同 id」，不是 id 本身
    identity_labels=True,
)


class PersonCard:
    spec = SPEC

    def build_dataset(self, target_samples: int, seed: int = DEFAULT_SEED) -> list[dict]:
        return build_person_dataset(target_samples=target_samples, seed=seed)

    def probe_units(self) -> list[ProbeUnit]:
        """共指任务没有「一个词 + 若干载体」这种单元，用句级指标验收（`evaluate_task`）。"""
        return []


CARD = PersonCard()
register(CARD)
