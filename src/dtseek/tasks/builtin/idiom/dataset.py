"""成语识别数据集：在句子里圈出所有成语。

任务形态：逐切片发射，一句话里 0 到 N 个成语，模型自己决定吐几个。
这是「可变切片数」最直接的检验 —— 没有固定锚点、没有 NMS。

语料来自 `idiom_examples`（由 `scripts/build_idiom_corpus.py` 从开源词典抽出），
**正负样本语体对齐**：例句是文学语体，背景句取自词典的出处引文与释义，
同样偏书面。这一点很关键 —— 若正面用文学语料、背景用口语对话，
模型会学成「口语 ⇒ 没有成语」这种语体捷径，而不是真的认识成语。

fail-closed 三条：
  - 标注覆盖句子里**所有**白名单成语（漏标一个就是错的监督信号）；
  - 成语之间不得重叠（重叠就无法判定标哪个，丢弃整句）；
  - 背景句必须**确实**一个成语都没有。
"""
from __future__ import annotations

import random

from dtseek.tasks.builtin.idiom.examples import BACKGROUND, EXAMPLES
from dtseek.tasks.builtin.idiom.lexicon import IDIOMS

# 类别 id 约定：0 背景，1 成语
NONE, IDIOM = 0, 1

# 只给「词典例句没覆盖到」的成语补保底样本；刻意做得杂，避免模型只认某一句式
SYNTH_FRAMES = (
    "老师让我们用{w}造一个句子。",
    "这篇作文里用到了{w}。",
    "他在本子上写下了{w}。",
    "字典里{w}这个词是什么意思？",
    "妈妈说他今天真是{w}。",
    "把{w}抄写三遍，明天听写。",
    "课文的第二段出现了{w}。",
    "请你在{w}下面画一条横线。",
    "读到这里，他突然明白了{w}的意思。",
    "老师在黑板上写下了{w}四个大字。",
    "{w}。",
    "{w}，这个词你认识吗？",
    "他把{w}读了两遍。",
    "这段话里{w}用得最妙。",
    "写作文的时候，他想起了一个词：{w}。",
    "同学们都在讨论{w}这个成语。",
)


def extract_idiom_spans(text: str) -> list[dict] | None:
    """找出句子里所有白名单成语的位置。

    返回 None 表示这一句**不可用**：存在互相重叠的成语匹配，
    标哪个都会漏掉另一个，属于无法标注的样本。
    """
    hits = [(i, text[i:i + 4]) for i in range(len(text) - 3) if text[i:i + 4] in IDIOMS]
    if not hits:
        return []
    chosen: list[tuple[int, str]] = []
    for i, w in hits:
        if any(i < j + 4 and j < i + 4 for j, _ in chosen):   # 与已选区间重叠
            return None
        chosen.append((i, w))
    chosen.sort()
    return [{"label": IDIOM, "word": w, "start": i, "end": i + 4} for i, w in chosen]


def _positives() -> list[dict]:
    """词典例句：标注句子里所有白名单成语（不止该条目的目标成语）。"""
    out, seen = [], set()
    for _target, text in EXAMPLES:
        if text in seen:
            continue
        spans = extract_idiom_spans(text)
        if not spans:
            continue
        seen.add(text)
        out.append({"text": text, "spans": spans})
    return out


def _synthesize_all(per_idiom_floor: int) -> list[dict]:
    """给**每个**成语都补若干模板句。

    早期只给「例句没覆盖到的成语」补保底，结果训练集全是文学例句、而实际输入/探针
    是白话短句，属于分布外：模型在模板句上经常「未触发」。现在每个成语都带模板样本，
    既补齐了每个成语的监督量，也让训练覆盖到白话句式。
    """
    out = []
    for idx, w in enumerate(sorted(IDIOMS)):
        for k in range(per_idiom_floor):
            # 用下标而不是 hash(w)：hash 受 PYTHONHASHSEED 影响，会破坏可复现性
            text = SYNTH_FRAMES[(idx + k) % len(SYNTH_FRAMES)].format(w=w)
            spans = extract_idiom_spans(text)
            if spans:
                out.append({"text": text, "spans": spans})
    return out


def build_idiom_dataset(target_samples: int = 9000, per_idiom_floor: int = 4,
                        bg_ratio: float = 0.35, seed: int = 20240927) -> list[dict]:
    """构建成语识别数据集。

    Args:
        target_samples: 目标总样本数
        per_idiom_floor: **每个**成语至少合成多少条模板句（保证无死角覆盖 + 覆盖白话句式）
        bg_ratio: 背景句占比（下界）
    """
    rng = random.Random(seed)
    real = _positives()
    synth = _synthesize_all(per_idiom_floor)
    positive = real + synth
    print(f"  词典例句 {len(real)} 条 + 每成语 {per_idiom_floor} 条模板句 {len(synth)} 条")

    n_bg = max(int(target_samples * bg_ratio), target_samples // 3)
    background = [s for s in BACKGROUND if extract_idiom_spans(s) == []]
    n_pos = min(len(positive), max(1, target_samples - n_bg))
    if n_bg > n_pos:
        # 正面样本只有词典例句这么多，背景不能反超 —— 否则模型学会「永远不开火」
        print(f"  正面样本仅 {n_pos} 条，背景配额由 {n_bg} 收到 {n_pos}（保持类别均衡）")
        n_bg = n_pos
    if len(background) < n_bg:
        raise ValueError(
            f"背景句只有 {len(background)} 条，不够 {n_bg} 条；背景太少模型会学成永远开火")
    rng.shuffle(background)
    rng.shuffle(positive)

    dataset = positive[:n_pos] + [{"text": s, "spans": []} for s in background[:n_bg]]
    rng.shuffle(dataset)

    cov = coverage_report(dataset)
    print(f"  成语识别数据集构建完成：{len(dataset)} 样本 "
          f"(含成语 {sum(1 for d in dataset if d['spans'])} / 背景 {sum(1 for d in dataset if not d['spans'])})"
          f"，覆盖成语 {cov['covered']}/{len(IDIOMS)} ✅")
    return dataset


def coverage_report(dataset: list[dict] | None = None) -> dict:
    """每个成语的样本量分布 —— 验证「无死角覆盖」。"""
    from collections import Counter
    if dataset is None:
        dataset = build_idiom_dataset()
    cnt: Counter = Counter()
    for item in dataset:
        for sp in item["spans"]:
            cnt[sp["word"]] += 1
    vals = sorted(cnt.values(), reverse=True)
    return {
        "n_idioms": len(IDIOMS),
        "covered": len(cnt),
        "missing": sorted(IDIOMS - set(cnt))[:20],
        "n_missing": len(IDIOMS - set(cnt)),
        "min": vals[-1] if vals else 0,
        "median": vals[len(vals) // 2] if vals else 0,
        "max": vals[0] if vals else 0,
    }


if __name__ == "__main__":
    ds = build_idiom_dataset(target_samples=3000, per_idiom_floor=2)
    for item in ds[:6]:
        print(f"  [{len(item['spans'])}] {item['text']} -> "
              f"{[(s['word'], s['start'], s['end']) for s in item['spans']]}")
    print(coverage_report(ds))
