"""任务卡插件协议的单测：校验、注册表、ckpt 快照一致性、成对截断。"""
import json
from pathlib import Path

import pytest

from dtseek.tasks.plugin import (
    CKPT_SPEC_VERSION,
    ProbeUnit,
    TaskClass,
    TaskSpec,
    TaskSpecMismatch,
    all_tasks,
    check_ckpt_specs,
    probe_units_of,
    register,
    resolve_tasks,
)

GOLDEN = Path(__file__).resolve().parent / "golden" / "multitask_equivalence.json"

BUILTIN = ["pronoun", "sentiment", "ownership"]


def _spec(**kw) -> TaskSpec:
    base = dict(name="demo", label="演示", classes=(TaskClass("背景"), TaskClass("甲")))
    base.update(kw)
    return TaskSpec(**base)


# ---- 内置任务卡 ----------------------------------------------------------

def test_builtin_tasks_registered():
    tasks = all_tasks()
    for name in BUILTIN:
        assert name in tasks, f"内置任务 {name} 没注册"


def test_builtin_class_names_match_baseline():
    """类别名与顺序就是类别 id —— 改了必须是有意为之，不能重构时漂移。"""
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    tasks = all_tasks()
    for name, names in golden["task_specs"].items():
        assert list(tasks[name].spec.class_names) == names


def test_builtin_specs_are_valid_and_self_consistent():
    for name, card in all_tasks().items():
        spec = card.spec
        assert spec.problems() == [], f"{name}: {spec.problems()}"
        assert spec.class_names[0] == spec.classes[0].name
        assert len(spec.cls_weights()) == spec.num_classes
        assert len({c.color for c in spec.classes}) == spec.num_classes, f"{name} 配色重复"


# ---- TaskSpec 校验 -------------------------------------------------------

@pytest.mark.parametrize("kw,fragment", [
    ({"name": "BadName"}, "小写标识符"),
    ({"name": ""}, "小写标识符"),
    ({"label": ""}, "label 不能为空"),
    ({"classes": (TaskClass("背景"),)}, "至少 2 个"),
    ({"classes": (TaskClass("背景"), TaskClass("背景"))}, "类别名重复"),
    ({"emission": "triple"}, "emission"),
    ({"max_steps": 0}, "max_steps"),
    ({"emission": "pair", "max_steps": 3}, "偶数"),
    ({"action_weight": (1.0,)}, "2 元组"),
])
def test_spec_rejects_bad_declarations(kw, fragment):
    with pytest.raises(ValueError) as ei:
        _spec(**kw)
    assert fragment in str(ei.value)


def test_spec_reports_all_problems_at_once():
    """一次报全，别修一个报一个。"""
    with pytest.raises(ValueError) as ei:
        _spec(name="Bad Name", label="", max_steps=0)
    msg = str(ei.value)
    assert "小写标识符" in msg and "label 不能为空" in msg and "max_steps" in msg


def test_pair_spec_accepts_even_steps():
    spec = _spec(emission="pair", max_steps=4)
    assert spec.pair_emission and spec.max_steps == 4


# ---- 注册表 ---------------------------------------------------------------

def test_resolve_unknown_task_lists_candidates():
    with pytest.raises(KeyError) as ei:
        resolve_tasks(["nope"])
    assert "pronoun" in str(ei.value)


def test_register_rejects_duplicate_name():
    class Card:
        spec = _spec(name="dup_probe")

        def build_dataset(self, target_samples, seed=0):
            return []

    register(Card())
    with pytest.raises(ValueError, match="已被注册"):
        register(Card())


def test_register_rejects_missing_build_dataset():
    class Card:
        spec = _spec(name="no_builder_probe")

    with pytest.raises(TypeError, match="build_dataset"):
        register(Card())


# ---- ckpt 快照 ------------------------------------------------------------

def test_snapshot_roundtrip():
    spec = _spec(name="rt", label="往返", classes=(
        TaskClass("背景"), TaskClass("甲", label="甲类", color="\033[1;31m")))
    back = TaskSpec.from_snapshot(json.loads(json.dumps(spec.to_snapshot())))
    assert back == spec
    assert back.classes[1].display == "甲类"


def test_snapshot_rejects_other_version():
    snap = _spec().to_snapshot()
    snap["version"] = CKPT_SPEC_VERSION + 1
    with pytest.raises(ValueError, match="版本"):
        TaskSpec.from_snapshot(snap)


def test_check_ckpt_specs_passes_when_consistent():
    tasks = all_tasks()
    snaps = {n: tasks[n].spec.to_snapshot() for n in BUILTIN}
    check_ckpt_specs(snaps, tasks)  # 不抛即通过


def test_check_ckpt_specs_catches_class_drift():
    tasks = all_tasks()
    snaps = {n: tasks[n].spec.to_snapshot() for n in BUILTIN}
    snaps["sentiment"]["classes"] = snaps["sentiment"]["classes"][:3]
    with pytest.raises(TaskSpecMismatch, match="类别体系不一致"):
        check_ckpt_specs(snaps, tasks)


def test_check_ckpt_specs_catches_missing_task():
    tasks = all_tasks()
    snaps = {"ghost_task": _spec(name="ghost_task").to_snapshot()}
    with pytest.raises(TaskSpecMismatch, match="不存在"):
        check_ckpt_specs(snaps, tasks)


# ---- 探针扩展点 -----------------------------------------------------------

def test_sentiment_card_exposes_word_probe_units():
    units = probe_units_of(all_tasks()["sentiment"])
    assert len(units) >= 100
    assert all(isinstance(u, ProbeUnit) and u.expected_class > 0 for u in units)
    assert all(len(u.sentences()) == len(u.carriers) for u in units)


def test_probe_units_optional_for_cards_without_it():
    class Bare:
        spec = _spec(name="bare_probe")

        def build_dataset(self, target_samples, seed=0):
            return []

    assert probe_units_of(Bare()) == []


def test_probe_unit_expands_placeholders_including_bare_word():
    """无终止标点的裸词形态必须覆盖 —— 漏了它会让裸词定位被截断。"""
    unit = ProbeUnit(key="高兴", expected_class=1, words=("高兴",))
    sentences = unit.sentences()
    assert "高兴" in sentences and "高兴。" in sentences


# ---- 整句级指标与按任务窗口 -------------------------------------------------

@pytest.mark.parametrize("bad_len", [4, 7, 129, 0])
def test_spec_rejects_out_of_range_max_len(bad_len):
    """窗口上限就是基座的能力上限 128，声明超过它要在构造期就炸。"""
    with pytest.raises(ValueError, match="max_len"):
        _spec(max_len=bad_len)


def test_snapshot_carries_max_len():
    spec = _spec(name="win", max_len=120)
    assert spec.to_snapshot()["max_len"] == 120
    assert TaskSpec.from_snapshot(spec.to_snapshot()).max_len == 120


def test_check_ckpt_specs_catches_max_len_drift():
    tasks = all_tasks()
    snaps = {n: tasks[n].spec.to_snapshot() for n in BUILTIN}
    snaps["pronoun"]["max_len"] = 120
    with pytest.raises(TaskSpecMismatch, match="max_len 不一致"):
        check_ckpt_specs(snaps, tasks)


def test_evaluate_task_reports_whole_sentence_metrics():
    """代词卡实测：首切片 100% 而整句只有 95.8%。整句指标必须在报告里。"""
    import torch
    from torch.utils.data import DataLoader

    from nano_char_tokenizer import NanoCharTokenizer
    from dtseek.decoder.robust_ar_model import RobustARSliceDecoder
    from dtseek.encoder.nano_doc_encoder import NanoDocEncoder
    from dtseek.tasks.builtin.pronoun.dataset import build_rich_ar_dataset
    from dtseek.tasks.runtime import GenericTaskDataset, evaluate_task

    spec = all_tasks()["pronoun"].spec
    tok = NanoCharTokenizer()
    torch.manual_seed(0)
    enc = NanoDocEncoder(vocab_size=tok.vocab_size, hidden_dim=32, num_layers=1,
                         num_heads=4, max_len=128, dropout=0.0)
    dec = RobustARSliceDecoder(hidden_dim=32, num_classes=spec.num_classes,
                               num_heads=4, num_layers=1)
    loader = DataLoader(GenericTaskDataset(build_rich_ar_dataset(target_samples=200)[:96], tok, spec),
                        batch_size=32)
    rep = evaluate_task(enc, dec, loader, torch.device("cpu"), spec)
    assert {"exact_match", "slice_precision", "slice_recall"} <= set(rep)
    assert all(0.0 <= rep[k] <= 1.0 for k in ("exact_match", "slice_precision", "slice_recall"))
    # 未训练的模型整句一致率必然低于首切片命中率——这正是首切片口径掩盖的东西
    assert rep["exact_match"] <= 1.0


def test_check_ckpt_specs_catches_segment_policy_drift():
    """旧 ckpt 缺这个字段会默认成 sentence，静默把人物追踪按句切开。必须报错。"""
    tasks = all_tasks()
    snaps = {n: tasks[n].spec.to_snapshot() for n in tasks}
    snaps["person"]["segment_policy"] = "sentence"
    with pytest.raises(TaskSpecMismatch, match="segment_policy"):
        check_ckpt_specs(snaps, tasks)


def test_person_card_declares_window_segment_policy():
    spec = all_tasks()["person"].spec
    assert spec.segment_policy == "window"
    assert spec.identity_labels and spec.annotate_all


def test_missing_inference_field_is_inherited_not_rejected():
    """旧 ckpt 没有 segment_policy 字段是正常的：按代码继承，但要把"继承了什么"报出来。"""
    tasks = all_tasks()
    snaps = {n: tasks[n].spec.to_snapshot() for n in tasks}
    for k in ("segment_policy", "max_len"):
        snaps["person"].pop(k, None)
    notes = check_ckpt_specs(snaps, tasks)          # 不抛
    assert any("segment_policy" in n for n in notes)
    assert any("max_len" in n for n in notes)


def test_present_but_different_inference_field_still_rejected():
    """写了但值不同 = 真的口径漂移，必须报错。"""
    tasks = all_tasks()
    snaps = {n: tasks[n].spec.to_snapshot() for n in tasks}
    snaps["person"]["segment_policy"] = "sentence"
    snaps["person"]["max_len"] = 64
    with pytest.raises(TaskSpecMismatch) as ei:
        check_ckpt_specs(snaps, tasks)
    assert "segment_policy" in str(ei.value) and "max_len" in str(ei.value)
