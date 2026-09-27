"""第三方任务卡通过 entry point 接入的真实验证。

这是插件标准的核心承诺：**不改 DTSeek 源码**，在自己的包里声明
`[project.entry-points."dtseek.tasks"]` 就能多出一张任务卡。
这里真的造一个装了 entry point 的临时发行包，走 `importlib.metadata` 的真实路径，
而不是 monkeypatch —— monkeypatch 只能证明我自己写的桩能跑。
"""
import sys
import textwrap

import pytest

import dtseek.tasks.plugin as tp
from dtseek.tasks.plugin import ENTRY_POINT_GROUP, TaskSpecMismatch, all_tasks, load_entry_point_tasks

PLUGIN_SOURCE = textwrap.dedent('''
    """第三方任务卡：不依赖 DTSeek 内部任何私有 API。"""
    from dtseek.tasks.plugin import ProbeUnit, TaskClass, TaskSpec

    SPEC = TaskSpec(
        name="third_party_demo",
        label="第三方演示",
        classes=(TaskClass("背景"), TaskClass("命中", color="\\033[1;35m")),
    )

    class Card:
        spec = SPEC

        def build_dataset(self, target_samples, seed=0):
            return [{"text": "这是一句话。", "spans": [{"label": 1, "start": 2, "end": 3}]}]

        def probe_units(self):
            return [ProbeUnit(key="一句", expected_class=1, words=("一句",))]

    CARD = Card()
''')

METADATA = "Metadata-Version: 2.1\nName: dtseek-third-party-demo\nVersion: 0.1\n"
ENTRY_POINTS = f"[{ENTRY_POINT_GROUP}]\nthird_party_demo = third_party_card:CARD\n"


@pytest.fixture
def installed_plugin(tmp_path, monkeypatch):
    """在临时目录里伪造一个「已安装」的发行包，含 entry point 与模块。"""
    (tmp_path / "third_party_card.py").write_text(PLUGIN_SOURCE, encoding="utf-8")
    dist = tmp_path / "dtseek_third_party_demo-0.1.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text(METADATA, encoding="utf-8")
    (dist / "entry_points.txt").write_text(ENTRY_POINTS, encoding="utf-8")

    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "third_party_card", raising=False)
    import importlib
    importlib.invalidate_caches()

    # 快照全局状态，测完精确还原，避免污染其它测试
    reg_before = dict(tp._REGISTRY)
    loaded_before = set(tp._PLUGINS_LOADED)
    yield tmp_path
    tp._REGISTRY.clear()
    tp._REGISTRY.update(reg_before)
    tp._PLUGINS_LOADED.clear()
    tp._PLUGINS_LOADED.update(loaded_before)


def test_entry_point_task_is_discovered_and_registered(installed_plugin):
    assert "third_party_demo" not in all_tasks(load_plugins=False)
    loaded = load_entry_point_tasks()
    assert "third_party_demo" in loaded
    assert "third_party_demo" in all_tasks()


def test_loaded_task_is_fully_usable(installed_plugin):
    load_entry_point_tasks()
    card = all_tasks()["third_party_demo"]
    data = card.build_dataset(10)
    assert data and data[0]["spans"][0]["label"] == 1
    assert card.spec.classes[1].display == "命中"
    assert tp.probe_units_of(card)[0].key == "一句"


def test_loading_is_idempotent(installed_plugin):
    """重复扫描不能因为「同名重复注册」炸掉。"""
    load_entry_point_tasks()
    load_entry_point_tasks()
    load_entry_point_tasks()
    assert list(all_tasks()).count("third_party_demo") == 1


def test_third_party_spec_lands_in_ckpt_snapshot(installed_plugin):
    """第三方卡的类别体系必须同样进 ckpt 快照并能通过一致性门禁。"""
    load_entry_point_tasks()
    tasks = all_tasks()
    snap = tasks["third_party_demo"].spec.to_snapshot()
    assert snap["name"] == "third_party_demo"
    tp.check_ckpt_specs({"third_party_demo": snap}, tasks)  # 不抛即通过

    # 类别改名 = 类别 id 语义变了，必须拦住
    renamed = dict(snap, classes=[dict(c) for c in snap["classes"]])
    renamed["classes"][1]["name"] = "被改过的类别"
    with pytest.raises(TaskSpecMismatch, match="类别体系不一致"):
        tp.check_ckpt_specs({"third_party_demo": renamed}, tasks)


def test_malformed_snapshot_is_reported_as_spec_mismatch(installed_plugin):
    """快照本身坏掉（类别数不足）也要走同一个错误出口，不漏出底层构造异常。"""
    load_entry_point_tasks()
    tasks = all_tasks()
    broken = dict(tasks["third_party_demo"].spec.to_snapshot())
    broken["classes"] = broken["classes"][:1]
    with pytest.raises(TaskSpecMismatch, match="无法解析"):
        tp.check_ckpt_specs({"third_party_demo": broken}, tasks)


def test_broken_third_party_card_fails_loudly(tmp_path, monkeypatch):
    """第三方声明不合法时必须在加载期炸掉，而不是静默带病注册。"""
    (tmp_path / "bad_card.py").write_text(
        "from dtseek.tasks.plugin import TaskClass, TaskSpec\n"
        "class Card:\n"
        "    spec = TaskSpec(name='bad', label='', classes=(TaskClass('背景'),))\n"
        "    def build_dataset(self, target_samples, seed=0):\n        return []\n"
        "CARD = Card()\n", encoding="utf-8")
    dist = tmp_path / "dtseek_bad-0.1.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: dtseek-bad\nVersion: 0.1\n", encoding="utf-8")
    (dist / "entry_points.txt").write_text(f"[{ENTRY_POINT_GROUP}]\nbad = bad_card:CARD\n", encoding="utf-8")

    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "bad_card", raising=False)
    import importlib
    importlib.invalidate_caches()

    reg_before = dict(tp._REGISTRY)
    loaded_before = set(tp._PLUGINS_LOADED)
    try:
        # spec 非法：TaskSpec 构造时就会抛，这是期望的 fail-closed 行为
        with pytest.raises(ValueError):
            load_entry_point_tasks()
    finally:
        tp._REGISTRY.clear()
        tp._REGISTRY.update(reg_before)
        tp._PLUGINS_LOADED.clear()
        tp._PLUGINS_LOADED.update(loaded_before)
