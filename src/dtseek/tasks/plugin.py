"""DTSeek 任务卡插件协议 —— 任务声明的**唯一**一点。

一个任务卡（TaskCard）把「类别体系 / 渲染配色 / 数据集构造 / 发射形态 / 探针单元」
声明在一处；训练、推理、验收三端都从这里读，不再各抄一份。

新增内置任务：
    1. 在 `dtseek/tasks/builtin/` 下开一个包，包里放一个 `spec: TaskSpec` +
       `build_dataset(...)` 的对象；
    2. 在 `dtseek/tasks/builtin/__init__.py` 里 import 一次。

新增第三方任务（不用改 DTSeek 源码）：
    在自己的包里声明 entry point:

        [project.entry-points."dtseek.tasks"]
        my_task = "my_pkg.my_task:CARD"

    然后 `all_tasks()` 默认会扫描到它。

为什么不把类别定义写在训练脚本里：
类别体系一旦有两个家（训练脚本 + 演示脚本），改一处漏一处就是静默的
「训练用 4 类、推理用 5 类」。所以这里的 `TaskSpec` 会随 ckpt 一起存快照，
加载时逐项校验，对不上直接抛错。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Protocol, runtime_checkable

ENTRY_POINT_GROUP = "dtseek.tasks"
CKPT_SPEC_VERSION = 1

#: 数据集构建器的默认随机种子（三个旧构建器历史上的固定值）
DEFAULT_SEED = 20240927

BACKGROUND_COLOR = "\033[90m"

#: 词/词对级探针的默认载体模板。`{w}` 是待测词，`{w2}` 是词对任务里的第二个词。
#: 必须同时覆盖「有终止标点」与「无终止标点」两种形态 —— 后者漏了会让裸词输入
#: 的定位被截断（情绪任务踩过这个坑）。
DEFAULT_PROBE_CARRIERS: tuple[str, ...] = (
    "{w}。",
    "说实话，{w}。",
    "现在就是{w}。",
    "刚看到这个消息，{w}。",
    "{w}",
    "真的很{w}",
)

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
EMISSION_MODES = ("single", "pair")


@dataclass(frozen=True)
class TaskClass:
    """类别体系里的一个类别。约定 **index 0 是背景类**（不发射切片）。"""

    name: str
    """类别名。既作训练监督的真值，也是 ckpt 快照里比对的值。"""

    label: str = ""
    """展示名，可以比 `name` 更啰嗦（如 name='积极' / label='积极/喜悦'）；留空则等于 name。"""

    color: str = BACKGROUND_COLOR
    """终端渲染的 ANSI 配色。"""

    @property
    def display(self) -> str:
        return self.label or self.name


@dataclass(frozen=True)
class ProbeUnit:
    """词级探针的最小单元：把待测词（对）填进若干载体句，逐个检查类别与定位。

    有了它，探针脚本不必认识任何具体任务；任务卡自己声明「我要考哪些词、期望是哪类」。
    """

    key: str
    """唯一标识，如 '高兴' 或 '高兴|难过'。"""

    expected_class: int
    """期望的类别 id（对应 TaskSpec.classes 的下标，必须 > 0）。"""

    words: tuple[str, ...]
    """填进载体的词。单切片任务 1 个（填 `{w}`），成对任务 2 个（`{w}` 与 `{w2}`）。"""

    carriers: tuple[str, ...] = DEFAULT_PROBE_CARRIERS

    def sentences(self) -> list[str]:
        """展开成待测句子。载体里多余的占位符按顺序绑定 words。"""
        out = []
        for tpl in self.carriers:
            try:
                out.append(tpl.format(w=self.words[0], w2=self.words[1] if len(self.words) > 1 else ""))
            except (IndexError, KeyError):
                out.append(tpl.format(w=self.words[0]))
        return out


@dataclass(frozen=True)
class TaskSpec:
    """任务卡的静态声明。这几个字段会写进 ckpt 快照，加载时逐项校验。"""

    name: str
    """稳定 id：作注册表键、ckpt 键，也是 `--tasks` 的取值。只允许小写字母/数字/下划线。"""

    label: str
    """中文展示名，如 '情绪倾向'。"""

    classes: tuple[TaskClass, ...]
    """类别体系。index 0 必须是背景类。"""

    max_steps: int = 4
    """单个句段最多发射几个切片。成对任务的切片数是 2 的倍数，故 max_steps 必须为偶数。"""

    max_len: int = 64
    """单个句段的编码窗口（字符数）。人物追踪这类需要跨多轮上下文的卡可以调到 120；
    基座本身支持到 128。窗口越大，段内可承载的上下文越多，但显存与耗时也越高。"""

    emission: str = "single"
    """single = 一个切片一个决策；pair = 连续两个切片构成一对（先左后右，按位置升序）。"""

    segment_policy: str = "sentence"
    """推理时怎么把长文切给这张卡。

    `sentence`（默认）：按标点切句，每句独立解码。适合"每个句子自成一体"的任务
    （代词/情绪/词义关系），它们的训练数据就是单句。
    `window`：**只在超出 `max_len` 时才切**，窗口内的多句一起解码。人物追踪必须用这个 ——
    跨句共指要求 id 在整段内保持一致，按句切开会让 id 每句从 1 重来，
    等于把任务毁掉（实测切句后漏标 62%、重复提及 id 全错）。

    ⚠ 这必须与训练口径一致：`GenericTaskDataset` 是按 `max_len` 整段编码的，
    所以 `window` 才是"训练怎么喂、推理就怎么切"。"""

    annotate_all: bool = False
    """样本里出现的切片**必须全部标注**，不允许被 `max_steps` 截断。

    默认 False（截断是"召回上限"，模型学会在 max_steps 处收束即可）。
    置 True 后 `GenericTaskDataset` 遇到超长样本会直接抛错 —— 因为"漏标一半 + 让模型
    在这里收束"是错的监督信号，必须转成响亮的失败，不能静默截断。"""

    identity_labels: bool = False
    """类别是**匿名身份槽**（人物1/人物2…）而不是语义类别时为 True。

    这类任务的 id 是任意分配的，所以真正要测的不是「id 猜得对不对」，而是
    「同一个人物的多次提及是否落在同一个 id 上」。置 True 后 `evaluate_task` 会额外
    报首次提及 / 重复提及 / 聚类一致性三项。"""

    cls_weight_bg: float = 1.3
    """背景类损失权重。背景句在自回归下只贡献一个监督信号，天然被稀释。"""

    span_weight: float = 1.5
    """起止指针损失权重。"""

    action_weight: tuple[float, float] = (0.8, 1.2)
    """(`<eos>`, `<cont>`) 的动作损失权重。稍偏 `<cont>` 以避免过早停机。"""

    def __post_init__(self) -> None:
        self.validate()

    # ---- 派生量 ----------------------------------------------------------

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def class_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.classes)

    @property
    def pair_emission(self) -> bool:
        return self.emission == "pair"

    def cls_weights(self) -> list[float]:
        return [self.cls_weight_bg] + [1.0] * (self.num_classes - 1)

    # ---- 校验 ------------------------------------------------------------

    def validate(self) -> None:
        problems = self.problems()
        if problems:
            raise ValueError(f"任务卡 {self.name!r} 声明不合法：" + "；".join(problems))

    def problems(self) -> list[str]:
        """返回全部违规项；空列表表示通过。一次报全，不修一个报一个。"""
        bad: list[str] = []
        if not self.name or not _NAME_RE.match(self.name):
            bad.append(f"name={self.name!r} 必须是非空的小写标识符")
        if not self.label:
            bad.append("label 不能为空")
        if len(self.classes) < 2:
            bad.append(f"classes 至少 2 个（背景 + 至少一个真类别），当前 {len(self.classes)}")
        names = [c.name for c in self.classes]
        if any(not n for n in names):
            bad.append("存在空类别名")
        if len(set(names)) != len(names):
            bad.append(f"类别名重复：{names}")
        if self.segment_policy not in ("sentence", "window"):
            bad.append(f"segment_policy={self.segment_policy!r} 只能是 sentence / window")
        if self.emission not in EMISSION_MODES:
            bad.append(f"emission={self.emission!r} 不在 {EMISSION_MODES}")
        if self.max_steps < 1:
            bad.append(f"max_steps={self.max_steps} 必须 >= 1")
        if not (8 <= self.max_len <= 128):
            bad.append(f"max_len={self.max_len} 必须落在 8..128（基座窗口上限 128）")
        if self.pair_emission and self.max_steps % 2 != 0:
            bad.append(f"成对任务的 max_steps 必须是偶数，当前 {self.max_steps}")
        if self.identity_labels and self.num_classes < 3:
            bad.append("identity_labels=True 至少要有 2 个身份槽（+背景类）")
        if len(self.action_weight) != 2:
            bad.append(f"action_weight 必须是 2 元组，当前 {self.action_weight!r}")
        return bad

    # ---- ckpt 快照 --------------------------------------------------------

    def to_snapshot(self) -> dict:
        """序列化成可进 ckpt 的纯 dict（推理端据此渲染，不再手抄第二份配色）。"""
        return {
            "version": CKPT_SPEC_VERSION,
            "name": self.name,
            "label": self.label,
            "classes": [{"name": c.name, "label": c.label, "color": c.color} for c in self.classes],
            "max_steps": self.max_steps,
            "max_len": self.max_len,
            "emission": self.emission,
            "segment_policy": self.segment_policy,
        }

    @staticmethod
    def from_snapshot(snap: dict) -> TaskSpec:
        ver = snap.get("version")
        if ver != CKPT_SPEC_VERSION:
            raise ValueError(
                f"ckpt 里的任务快照版本是 {ver!r}，本代码只认识 {CKPT_SPEC_VERSION}。"
                " 请用当前 training/train_multitask.py 重新训练。")
        return TaskSpec(
            name=snap["name"],
            label=snap["label"],
            classes=tuple(TaskClass(name=c["name"], label=c.get("label", ""),
                                    color=c.get("color", BACKGROUND_COLOR))
                          for c in snap["classes"]),
            max_steps=snap["max_steps"],
            max_len=snap.get("max_len", 64),
            emission=snap["emission"],
            segment_policy=snap.get("segment_policy", "sentence"),
        )


@runtime_checkable
class TaskCard(Protocol):
    """任务卡：一份静态声明 + 一个数据集构造器（+ 可选的探针单元）。"""

    spec: TaskSpec

    def build_dataset(self, target_samples: int, seed: int = DEFAULT_SEED) -> list[dict]:
        """返回 `[{text, spans:[{label,start,end}]}]`。

        - `spans` 为空 = 背景样本；
        - `label` 是 `spec.classes` 的下标，`start`/`end` 是 **0-based 半开区间**；
        - 同一个样本的 spans 必须能按 start 升序排成最终发射顺序（成对任务即左、右交替）。
        """
        ...

    def probe_units(self) -> list[ProbeUnit]:
        """可选的词级探针单元。没有就不实现 —— 用 `probe_units_of()` 取，取不到返回空。"""
        ...


class TaskSpecMismatch(ValueError):
    """ckpt 快照与当前注册表对不上 —— 静默跑下去只会得出错的结果。"""


_REGISTRY: dict[str, TaskCard] = {}
_PLUGINS_LOADED: set[str] = set()


def register(card: TaskCard) -> TaskCard:
    """注册一张任务卡。重复注册同名但不同对象会抛错（防止两处定义悄悄分叉）。"""
    spec = getattr(card, "spec", None)
    if not isinstance(spec, TaskSpec):
        raise TypeError(f"{card!r} 没有合法的 spec: TaskSpec 属性")
    spec.validate()
    if not callable(getattr(card, "build_dataset", None)):
        raise TypeError(f"任务卡 {spec.name!r} 缺少 build_dataset()")
    existing = _REGISTRY.get(spec.name)
    if existing is not None and existing is not card:
        raise ValueError(f"任务名 {spec.name!r} 已被注册，不允许覆盖")
    _REGISTRY[spec.name] = card
    return card


def _ensure_builtins() -> None:
    import dtseek.tasks.builtin  # noqa: F401  导入即注册内置卡


def load_entry_point_tasks() -> list[str]:
    """扫描 entry point 注册的第三方任务卡。幂等，可以反复调。"""
    loaded = []
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        if ep.name in _PLUGINS_LOADED:
            continue
        card = ep.load()
        register(card)
        _PLUGINS_LOADED.add(ep.name)
        loaded.append(card.spec.name)
    return loaded


def all_tasks(*, load_plugins: bool = True) -> dict[str, TaskCard]:
    """全部已注册任务卡（内置的 + entry point 来的），按注册顺序。"""
    _ensure_builtins()
    if load_plugins:
        load_entry_point_tasks()
    return dict(_REGISTRY)


def get_task(name: str) -> TaskCard:
    tasks = all_tasks()
    if name not in tasks:
        raise KeyError(f"未注册的任务 {name!r}；已注册：{sorted(tasks)}")
    return tasks[name]


def resolve_tasks(names: list[str] | None = None) -> dict[str, TaskCard]:
    """按名字取子集；names 为 None 时返回全部。名字拼错会抛 KeyError 并列出候选。"""
    tasks = all_tasks()
    if names is None:
        return tasks
    missing = [n for n in names if n not in tasks]
    if missing:
        raise KeyError(f"未注册的任务 {missing}；已注册：{sorted(tasks)}")
    return {n: tasks[n] for n in names}


def probe_units_of(card: TaskCard) -> list[ProbeUnit]:
    """取任务卡的探针单元；没实现该扩展点就返回空列表。"""
    fn = getattr(card, "probe_units", None)
    if not callable(fn):
        return []
    return list(fn())


#: 快照里**可以缺**、缺了就按代码当前值继承的字段。
#: 它们都是推理行为（怎么切段、按多大窗口喂），不改变权重含义 ——
#: 旧 ckpt 没这些字段是正常的，按代码走才是对的。
INHERITABLE_KEYS = ("max_len", "segment_policy", "identity_labels", "annotate_all")


def check_ckpt_specs(snapshots: dict[str, dict], cards: dict[str, TaskCard]) -> list[str]:
    """加载 ckpt 前的一致性门禁（fail-closed）。返回"从代码继承"的字段说明，供上层提示。

    逐项校验，任一不合就抛 `TaskSpecMismatch`：
      - 快照里的每个任务都必须有对应的已注册任务卡
      - 类别名列表必须**逐字相同**（顺序也相同 —— 顺序即类别 id）
      - `max_steps` / `emission` 必须相同
      - 快照里**确实写了**的推理行为字段（`INHERITABLE_KEYS`）必须与代码一致

    为什么要把"缺字段"和"值不同"分开：
      - 缺字段 = ckpt 早于该字段诞生，代码里的值就是当前行为 → 继承 + 提示；
      - 值不同 = 真的口径漂移。静默按错的那个跑会让推理与训练口径不一致
        （实测人物追踪整句一致率 16.2% → 0.8%），必须报错。
    """
    problems: list[str] = []
    inherited: list[str] = []
    for name, snap in snapshots.items():
        card = cards.get(name)
        if card is None:
            problems.append(f"ckpt 里的任务 {name!r} 在当前注册表中不存在")
            continue
        try:
            ckpt_spec = TaskSpec.from_snapshot(snap)
        except (ValueError, KeyError, TypeError) as exc:
            # 快照本身坏了（版本不符 / 字段缺失 / 声明非法）也归到同一类错误，
            # 让"加载即报错"只有一个出口，而不是漏出底层构造异常。
            problems.append(f"任务 {name!r} 的 ckpt 快照无法解析：{exc}")
            continue
        cur = card.spec
        for key in INHERITABLE_KEYS:
            if key not in snap:
                inherited.append(f"{name}.{key} 快照里没有该字段，按代码当前值 {getattr(cur, key)!r} 继承")
        if ckpt_spec.class_names != cur.class_names:
            problems.append(
                f"任务 {name!r} 类别体系不一致：ckpt={list(ckpt_spec.class_names)} vs 代码={list(cur.class_names)}")
        if ckpt_spec.max_steps != cur.max_steps:
            problems.append(f"任务 {name!r} max_steps 不一致：ckpt={ckpt_spec.max_steps} vs 代码={cur.max_steps}")
        if ckpt_spec.emission != cur.emission:
            problems.append(f"任务 {name!r} emission 不一致：ckpt={ckpt_spec.emission} vs 代码={cur.emission}")
        if "max_len" in snap and ckpt_spec.max_len != cur.max_len:
            problems.append(f"任务 {name!r} max_len 不一致：ckpt={ckpt_spec.max_len} vs 代码={cur.max_len}")
        if "segment_policy" in snap and ckpt_spec.segment_policy != cur.segment_policy:
            # 这条最阴：旧 ckpt 快照没有这个字段会默认成 sentence，
            # 于是人物追踪在推理时被按句切开、id 每句从 1 重来 —— 实测整句一致率
            # 从 16.2% 掉到 0.8%。写了但写错，必须报错。
            problems.append(
                f"任务 {name!r} segment_policy 不一致：ckpt={ckpt_spec.segment_policy} vs 代码={cur.segment_policy}")
    if problems:
        raise TaskSpecMismatch("ckpt 与当前任务声明对不上：\n  - " + "\n  - ".join(problems))
    return inherited
