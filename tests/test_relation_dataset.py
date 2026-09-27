"""近义/反义成对标注数据集的 fail-closed 校验。

重点验三件事：
  1. 句子里出现**另一对完整的词**时必须被拦住（这是会让监督信号出错的那种污染）
  2. 定位必须唯一且不重叠，否则丢弃样本
  3. 成对任务的切片数恒为偶数、每对词都有覆盖下限
"""
import pytest

from dtseek.tasks.builtin.relation.lexicon import ANTONYM_PAIRS, SYNONYM_PAIRS
from dtseek.tasks.builtin.relation.dataset import (
    ALL_PAIRS,
    build_relation_dataset,
    find_unannotated_pair,
    lexicon_words_in,
    locate_pair,
    pair_coverage_report,
)
from dtseek.tasks.plugin import all_tasks
from dtseek.tasks.runtime import GenericTaskDataset

PAIR_A = SYNONYM_PAIRS[0]
PAIR_B = ANTONYM_PAIRS[0]


# ---- 未标注词对检测 -------------------------------------------------------

def test_detects_sentence_containing_both_members_of_a_pair():
    a, b = PAIR_A
    assert find_unannotated_pair(f"他在{a}和{b}之间犹豫。") == (a, b)


def test_allowed_pair_is_not_flagged_in_either_order():
    a, b = PAIR_A
    assert find_unannotated_pair(f"{a}和{b}。", allowed=(a, b)) is None
    assert find_unannotated_pair(f"{b}和{a}。", allowed=(a, b)) is None


def test_single_member_alone_is_not_flagged():
    """单个词出现不算词对 —— 这是有用的干扰项，不该拦。"""
    a, _ = PAIR_A
    assert find_unannotated_pair(f"他提到了{a}这个词。") is None


def test_lexicon_words_in_finds_lengths_2_to_4():
    a, b = PAIR_A
    assert {a, b} <= lexicon_words_in(f"{a}和{b}。")


# ---- 定位 ----------------------------------------------------------------

def test_locate_pair_returns_sorted_spans():
    spans = locate_pair("他既高兴又难过。", "高兴", "难过")
    assert spans is not None
    assert [s["word"] for s in spans] == ["高兴", "难过"]
    assert spans[0]["start"] < spans[1]["start"]
    assert "他既高兴又难过。"[spans[1]["start"]:spans[1]["end"]] == "难过"


def test_locate_pair_rejects_missing_or_ambiguous():
    assert locate_pair("这句话里没有那两个词。", "高兴", "难过") is None
    # 同一个词出现两次 ⇒ 无法判定指哪一个
    assert locate_pair("高兴就是高兴。", "高兴", "难过") is None


# ---- 数据集整体不变量 -----------------------------------------------------

@pytest.fixture(scope="module")
def dataset() -> list[dict]:
    return build_relation_dataset(target_samples=600, per_pair_floor=1, per_pair_cap=2)


def test_every_sample_has_even_slice_count(dataset):
    """成对任务吐奇数个切片就是配对错位 —— 数据集层面先保证真值是偶数。"""
    for item in dataset:
        assert len(item["spans"]) % 2 == 0, item


def test_positive_samples_have_exactly_one_pair_with_consistent_label(dataset):
    for item in dataset:
        if not item["spans"]:
            continue
        assert len(item["spans"]) == 2, item
        labels = {s["label"] for s in item["spans"]}
        assert labels == {item["category"]}, item
        assert item["category"] in (1, 2), item


def test_spans_actually_point_at_the_words(dataset):
    for item in dataset:
        for s in item["spans"]:
            assert item["text"][s["start"]:s["end"]] == s["word"], item


def test_no_sample_contains_a_second_unannotated_pair(dataset):
    for item in dataset:
        allowed = tuple(s["word"] for s in item["spans"]) or None
        assert find_unannotated_pair(item["text"], allowed=allowed) is None, item


def test_background_samples_are_empty_and_clean(dataset):
    bg = [d for d in dataset if not d["spans"]]
    assert bg, "没有背景样本，模型会学成永远开火"
    for item in bg:
        assert find_unannotated_pair(item["text"]) is None, item


def test_every_declared_pair_gets_floor_coverage(dataset):
    """Zipf 长尾的教训：只报整体准确率，会掩盖「某些词对从没被训过」。"""
    rep = pair_coverage_report(dataset)
    assert rep["n_pairs_declared"] == len(ALL_PAIRS)
    assert rep["n_missing"] == 0, f"以下词对没有任何样本：{rep['missing']}"


# ---- 与任务卡/运行时的接缝 ------------------------------------------------

def test_relation_card_is_registered_and_paired():
    card = all_tasks()["relation"]
    assert card.spec.pair_emission
    assert card.spec.max_steps % 2 == 0
    assert list(card.spec.class_names) == ["无关系", "近义词", "反义词"]


def test_generic_dataset_truncates_pairs_whole(dataset):
    """max_steps 截断时不能把一对切成半个。"""
    from nano_char_tokenizer import NanoCharTokenizer
    card = all_tasks()["relation"]
    spec = card.spec
    item = next(d for d in dataset if len(d["spans"]) == 2)
    item3 = {**item, "spans": item["spans"] + [{"word": "占位", "label": 1, "start": 0, "end": 1}]}
    ds = GenericTaskDataset([item3], NanoCharTokenizer(), spec, max_steps=4)
    row = ds[0]
    assert row["step_mask"].sum().item() == 2, "3 个切片应被截成 2 个（成对），不能是 3 个"
