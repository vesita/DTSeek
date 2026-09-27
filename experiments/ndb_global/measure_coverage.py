"""上界测量：换帧探针词表里有多少词/词对在训练语料中从未出现过？

只读 src/ 的既有实现，不改任何东西。NDB 唯一能多救的就是「没见过的词」，
所以这个差集直接决定了全局 NDB 有没有存在意义。

口径（两种，都报）：
  A. 词表覆盖（NDB 能不能查到这条键）—— 探针词是否作为**标注正例**出现在训练集中。
     NDB 只回写模型自己发射过的切片，所以只有被标为正例的词才会让键有值。
  B. 字面覆盖（更宽）—— 探针词是否在训练集任意文本里出现过（含背景/别的词当子串）。
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dtseek.tasks.builtin.idiom.dataset import build_idiom_dataset  # noqa: E402
from dtseek.tasks.builtin.idiom.lexicon import IDIOMS as IDIOM_LEX  # noqa: E402
from dtseek.tasks.builtin.relation.dataset import (  # noqa: E402
    build_relation_dataset, ALL_PAIRS,
)
from dtseek.tasks.plugin import all_tasks, probe_units_of  # noqa: E402

SEED = 42
IDIOM_SAMPLES = 14000     # training/train_task_card.py 的默认值
RELATION_SAMPLES = 14000


def train_split(data: list[dict]) -> tuple[list[dict], list[dict]]:
    """复刻 train_task_card.py 的切分：random.Random(seed).shuffle + 前 10% 当验证。"""
    data = list(data)
    random.Random(SEED).shuffle(data)
    n_val = max(200, len(data) // 10)
    return data[n_val:], data[:n_val]


def positive_labels(data: list[dict]) -> Counter:
    """训练集里作为**标注正例**出现过的词 -> 次数。"""
    c: Counter = Counter()
    for item in data:
        for sp in item.get("spans", []):
            c[sp["word"]] += 1
    return c


def literal_presence(data: list[dict], words: set[str]) -> set[str]:
    """字面在任意样本 text 里出现过的词（暴力找子串，句子都很短）。"""
    hits: set[str] = set()
    for item in data:
        t = item["text"]
        for w in words:
            if w in t:
                hits.add(w)
    return hits


def units_of(card: str) -> list:
    return probe_units_of(all_tasks()[card])


def report_idiom() -> dict:
    print("=" * 78)
    print("IDIOM  —— 构建训练集（只调用既有 build_idiom_dataset）")
    print("=" * 78)
    data = build_idiom_dataset(target_samples=IDIOM_SAMPLES)
    train, val = train_split(data)
    covered_all = set(positive_labels(data))
    covered_train = set(positive_labels(train))

    units = units_of("idiom")
    probe_words = {u.key for u in units}
    print(f"\n  探针单元（成语）数: {len(units)}")
    print(f"  词表白名单 IDIOMS : {len(IDIOM_LEX)}")
    print(f"  训练集(全体 {len(data)} 条) 标注覆盖: {len(covered_all)}/{len(IDIOM_LEX)}")
    print(f"  训练集(训练切分 {len(train)} 条) 标注覆盖: {len(covered_train)}/{len(IDIOM_LEX)}")

    never_all = sorted(probe_words - covered_all)
    never_train = sorted(probe_words - covered_train)
    print(f"\n  探针词里，在**全体数据**中从未被标为正例的: {len(never_all)}"
          f"  ({len(never_all)/max(1,len(probe_words))*100:.1f}%)")
    print(f"  探针词里，在**训练切分**中从未被标为正例的: {len(never_train)}"
          f"  ({len(never_train)/max(1,len(probe_words))*100:.1f}%)")
    if never_train[:10]:
        print(f"    例: {never_train[:10]}")

    # 字面覆盖（更宽口径）
    lit_train = literal_presence(train, probe_words)
    never_lit = sorted(probe_words - lit_train)
    print(f"  探针词里，字面在训练切分任意文本中从未出现的: {len(never_lit)}"
          f"  ({len(never_lit)/max(1,len(probe_words))*100:.1f}%)")
    if never_lit[:10]:
        print(f"    例: {never_lit[:10]}")

    return {
        "task": "idiom",
        "n_probe_units": len(units),
        "n_lexicon": len(IDIOM_LEX),
        "n_dataset": len(data),
        "covered_all": len(covered_all),
        "covered_train": len(covered_train),
        "never_labeled_all": never_all,
        "never_labeled_train": never_train,
        "never_literal_train": never_lit,
    }


def report_relation() -> dict:
    print()
    print("=" * 78)
    print("RELATION —— 构建训练集（只调用既有 build_relation_dataset）")
    print("=" * 78)
    data = build_relation_dataset(target_samples=RELATION_SAMPLES)
    train, val = train_split(data)

    pair_all = {d["pair_key"] for d in data if d.get("pair_key")}
    pair_train = {d["pair_key"] for d in train if d.get("pair_key")}

    units = units_of("relation")
    probe_pairs = [u.key for u in units]
    probe_words = {w for u in units for w in u.words}

    declared_pairs = {f"{a}|{b}" for a, b in ALL_PAIRS}
    print(f"\n  探针单元（词对）数: {len(units)}   词表声明对: {len(declared_pairs)}")
    print(f"  训练集(全体 {len(data)} 条) 覆盖词对: {len(pair_all)}/{len(declared_pairs)}")
    print(f"  训练集(训练切分 {len(train)} 条) 覆盖词对: {len(pair_train)}/{len(declared_pairs)}")

    pp = set(probe_pairs)
    never_pair_all = sorted(pp - pair_all)
    never_pair_train = sorted(pp - pair_train)
    print(f"\n  探针词对里，在**全体数据**中从未被标为正例的: {len(never_pair_all)}"
          f"  ({len(never_pair_all)/max(1,len(pp))*100:.1f}%)")
    print(f"  探针词对里，在**训练切分**中从未被标为正例的: {len(never_pair_train)}"
          f"  ({len(never_pair_train)/max(1,len(pp))*100:.1f}%)")
    if never_pair_train[:10]:
        print(f"    例: {never_pair_train[:10]}")

    labeled_words_train = {sp["word"] for d in train for sp in d.get("spans", [])}
    never_word_train = sorted(probe_words - labeled_words_train)
    lit_train = literal_presence(train, probe_words)
    never_lit = sorted(probe_words - lit_train)
    print(f"\n  探针**词**（去重 {len(probe_words)} 个）里，训练切分中从未被标为正例的: "
          f"{len(never_word_train)} ({len(never_word_train)/max(1,len(probe_words))*100:.1f}%)")
    print(f"  探针**词**里，字面在训练切分任意文本中从未出现的: {len(never_lit)}"
          f"  ({len(never_lit)/max(1,len(probe_words))*100:.1f}%)")
    if never_lit[:10]:
        print(f"    例: {never_lit[:10]}")

    return {
        "task": "relation",
        "n_probe_pairs": len(pp),
        "n_probe_words": len(probe_words),
        "n_declared_pairs": len(declared_pairs),
        "n_dataset": len(data),
        "covered_pairs_all": len(pair_all),
        "covered_pairs_train": len(pair_train),
        "never_pair_all": never_pair_all,
        "never_pair_train": never_pair_train,
        "never_word_train": never_word_train,
        "never_literal_train": never_lit,
    }


if __name__ == "__main__":
    out = {"idiom": report_idiom(), "relation": report_relation()}
    dest = Path(__file__).parent / "coverage_result.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n结果已写 {dest}")
