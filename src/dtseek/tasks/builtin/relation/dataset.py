"""近义词 / 反义词成对标注数据集。

任务形态：一句话里同时出现一对词，模型按「左、右」顺序连续发射两个切片，
两个切片共享同一个类别（近义词对 / 反义词对）。**成对关系由发射顺序隐式表达**，
解码器不用改 —— 这是先试隐式配对的理由。

设计上的三处关键取舍：

1. **关系类型不能由句式推断。**
   同一个模板必须同时用于近义词对和反义词对，否则模型学到的是
   「这个句式 → 这个类别」的捷径，而不是两个词本身的语义关系。
   因此模板池以**关系中立**的句式为主（"…和…这两个词要一起记"），
   少数"关系倾向"句式（并列 / 转折）也**成对使用**、不偏向任何一类。
   验收时必须另加**换帧探针**：只用中立帧训练、用没见过的帧测，才能暴露捷径。

2. **词表里每个词对都要有覆盖下限。**
   情绪任务踩过 Zipf 长尾的坑：句级 98% 而词级只有 43.7%。
   这里同样给每对词设 `per_pair_floor`（保底）与 `per_pair_cap`（压长尾）。

3. **一句里不能出现第二个完整的词对。**
   "帧模板不含任何词表词"是过严的规则 —— 单个词（如"一起"）单独出现并不构成
   一个词对，反而是有用的干扰项。真正会让监督信号出错的是：句子**同时**含
   `a` 和 `b` 两个词、而这对 `(a, b)` 没被标注。所以校验按「完整词对」做，
   逐样本检查，命中就丢弃该样本。
"""
from __future__ import annotations

import glob
import random
import re

from dtseek.tasks.builtin.relation.lexicon import ANTONYM_PAIRS, SYNONYM_PAIRS

# 类别 id 约定：0 背景，1 近义词对，2 反义词对
NONE, SYNONYM, ANTONYM = 0, 1, 2

PAIR_TABLES = {SYNONYM: SYNONYM_PAIRS, ANTONYM: ANTONYM_PAIRS}
ALL_PAIRS = [tuple(p) for pairs in PAIR_TABLES.values() for p in pairs]

VOCAB: set[str] = {w for a, b in ALL_PAIRS for w in (a, b)}
_WORD_LENGTHS = (2, 3, 4)

# ── 关系中立帧（主体）：句式本身不含任何近义/反义线索 ──────────────────────
#
# 帧数必须够多、句式必须够杂。实测教训：只有 14 个帧时，句级「对完整命中率」能到
# 97.9%，但换帧探针只有 25.4% —— 模型背下的是「这个句式里两个词在哪个位置」，
# 不是两个词的关系。句长、词的位置、有没有上下文，都要有变化。
NEUTRAL_FRAMES = [
    # 短句 / 裸词（真实输入常见形态；漏了它，短句上的定位会整体崩）
    "{w}和{w2}。",
    "{w}、{w2}。",
    "{w}和{w2}这两个词要一起记。",
    # 课堂场景
    "老师让我们区分{w}和{w2}。",
    "老师把{w}和{w2}写在黑板上。",
    "老师让他用{w}和{w2}各说一句话。",
    "昨天的听写里，{w}和{w2}他都写对了。",
    "请在{w}和{w2}下面画横线。",
    "作业要求给{w}和{w2}各造一个句子。",
    "我们一起来读一读{w}和{w2}。",
    "{w}和{w2}，老师讲过好几遍了。",
    "这次考试的题目里有{w}和{w2}。",
    "{w}和{w2}的区别，你能说清楚吗？",
    "请说说{w}和{w2}的区别。",
    # 读书 / 写作场景
    "课文里出现了{w}和{w2}。",
    "这本书里{w}和{w2}都出现过。",
    "读课文的时候遇到{w}和{w2}，要圈出来。",
    "这段话里的{w}和{w2}用得都很恰当。",
    "作文里同时用了{w}和{w2}，老师夸了他。",
    "这段文章用了{w}和{w2}来对比。",
    "看到{w}和{w2}，他就想起了那篇课文。",
    "读到{w}和{w2}时，他停下来做了批注。",
    "他查了字典，弄明白了{w}和{w2}。",
    # 日常 / 口语场景
    "他把{w}写成了{w2}。",
    "小明分不清{w}和{w2}。",
    "他分不清{w}和{w2}，就来问我。",
    "他把{w}和{w2}记在了同一个本子上。",
    "他把{w}、{w2}两个词抄了五遍。",
    "这两个词{w}、{w2}他都会写。",
    "{w}、{w2}，这两个词你认识吗？",
    "妈妈问我{w}和{w2}的意思有什么差别。",
    "妹妹不认识{w}，也不认识{w2}。",
    "{w}和{w2}之间到底有什么不同？",
    "他第一次听到{w}和{w2}是在同一篇课文里。",
    "{w}和{w2}这两个词经常被放在一起讲。",
    "路上他一直在琢磨{w}和{w2}。",
]

# ── 关系倾向帧：并列暗示近义、转折暗示反义，但两类都各用一半 ────────────────
LEAN_FRAMES = {
    SYNONYM: [
        "他既{w}又{w2}。",
        "这个人很{w}，也很{w2}。",
    ],
    ANTONYM: [
        "他不是{w}，而是{w2}。",
        "这里{w}，那里却{w2}。",
    ],
}

# ── 上下文前后缀：让正例的**句长**不携带类别信息 ──────────────────────────
#
# 实测教训：正例全是「两个四字成语 + 连接词」的 12~20 字短句时，模型学会的是
# 「短句 ⇒ 有词对」——12 条中性短句上误报 50%，而 6 条长句上误报 0%。
# 句长必须和背景样本同分布，否则它就是一条捷径。
CONTEXT_PREFIXES = (
    "昨天的语文课上，", "他想了很久，", "翻开课本第三页，", "老师讲到这里的时候，",
    "妈妈说，", "读完整段话，", "考试的时候，", "他不假思索地写道，",
    "这件事过去很多年了，", "整理旧笔记时，",
)
CONTEXT_SUFFIXES = (
    "，老师点了点头。", "，同学们都记住了。", "，这个问题就解决了。",
    "，他觉得自己进步了不少。", "，回家后还专门查了字典。", "，这篇课文就讲完了。",
    "，直到现在他还记得很清楚。", "，我在旁边听着也觉得有道理。",
)


# 背景句池：同样按「不含完整词对」校验
BACKGROUND_SENTENCES = [
    "今天的作业是把生字抄三遍。",
    "操场上同学们正在跑步。",
    "妈妈买回来两个又大又红的苹果。",
    "春天到了，柳树发出了新芽。",
    "他每天早上七点起床。",
    "教室里安静得能听见笔尖的声音。",
    "老师说下周一要交读书笔记。",
    "雨停了，天边出现了一道彩虹。",
    "我把书包整理好放在椅子上。",
    "小河从村口缓缓流过。",
    "爷爷在院子里种了几棵向日葵。",
    "下课铃响了，大家跑向操场。",
    "他把课本翻到第十二页。",
    "窗外的树叶被风吹得沙沙响。",
    "我们一共种了二十四棵树苗。",
    "这道题他想了很久才做出来。",
    "图书馆里要保持安静。",
    "妹妹把画贴在了墙上。",
    "秋天到了，田里的稻子黄了。",
    "他背着书包走进了校门。",
]


def lexicon_words_in(text: str) -> set[str]:
    """扫出句子里出现的所有词表词。句子 <=64 字，暴力枚举 2/3/4 字子串即可。"""
    hits = set()
    for i in range(len(text)):
        for length in _WORD_LENGTHS:
            w = text[i:i + length]
            if w in VOCAB:
                hits.add(w)
    return hits


def find_unannotated_pair(text: str, allowed: tuple[str, str] | None = None) -> tuple[str, str] | None:
    """返回句子里出现的**第一个未被标注的完整词对**；没有则返回 None。

    `allowed` 用集合比较而不是元组比较 —— 插入时词序可能被交换过，
    `(a, b)` 与 `(b, a)` 是同一对。
    """
    hits = lexicon_words_in(text)
    allowed_key = frozenset(allowed) if allowed else None
    for a, b in ALL_PAIRS:
        if allowed_key is not None and frozenset((a, b)) == allowed_key:
            continue
        if a in hits and b in hits:
            return (a, b)
    return None


def locate_pair(text: str, a: str, b: str) -> list[dict] | None:
    """在句子里定位这一对词。任一找不到 / 重叠 / 出现多次 → 返回 None（丢弃该样本）。

    fail-closed：宁可少一条样本，也不要一条起点算错的监督信号。
    """
    ia, ib = text.find(a), text.find(b)
    if ia < 0 or ib < 0:
        return None
    if text.find(a, ia + 1) >= 0 or text.find(b, ib + 1) >= 0:
        return None                      # 同一个词出现多次，无法判定指哪一个
    if (ia < ib and ia + len(a) > ib) or (ib < ia and ib + len(b) > ia):
        return None                      # 两个字面重叠
    spans = [{"word": a, "start": ia, "end": ia + len(a)},
             {"word": b, "start": ib, "end": ib + len(b)}]
    spans.sort(key=lambda s: s["start"])
    return spans


def audit_templates() -> dict:
    """静态体检：哪些帧/背景句在某些词对上会撞出未标注的词对。

    只用于构建时打印告警 —— 真正的拦截在 `_make_sample` 里逐样本做。
    """
    report = {"frames": {}, "background": []}
    for frame in NEUTRAL_FRAMES + LEAN_FRAMES[SYNONYM] + LEAN_FRAMES[ANTONYM]:
        probe = frame.format(w="高兴", w2="难过")
        if find_unannotated_pair(probe) is not None:
            report["frames"][frame] = find_unannotated_pair(probe)
    for s in BACKGROUND_SENTENCES:
        bad = find_unannotated_pair(s)
        if bad:
            report["background"].append((s, bad))
    for ctx in CONTEXT_PREFIXES + CONTEXT_SUFFIXES:
        bad = find_unannotated_pair(ctx)
        if bad:
            report["background"].append((ctx, bad))
    return report


CORPUS_GLOB = "/home/vesita/coding/my/nanoSeek/data/chinese/*dialogue.txt"
ROLE_PREFIX = re.compile(r"^(用户|模型|系统|提问|回答|User|Assistant)[:：]\s*")
SENT_SPLIT = re.compile(r"[。！？\n；;]+")


def _mine_neutral(corpus_glob: str, n: int, max_len: int = 64,
                  max_lines: int | None = None) -> list[str]:
    """从真实语料里挖中性句（不含任何词表词）当背景。

    为什么不能只用那 20 条手写背景句：实测在 12 条普通中性句上误报 8%，
    而验证集报 0.0% —— 因为背景池只有 20 句，bg_fp 根本没被测出来。
    背景必须来自真实分布的文本，验证才有意义。
    """
    pool: list[str] = []
    lines = 0
    for path in sorted(glob.glob(corpus_glob)):
        with open(path, encoding="utf-8", errors="ignore") as fp:
            for line in fp:
                lines += 1
                if max_lines is not None and lines > max_lines:
                    return pool
                text = ROLE_PREFIX.sub("", line.strip())
                for s in SENT_SPLIT.split(text):
                    s = s.strip()
                    if 4 <= len(s) <= max_len and not lexicon_words_in(s):
                        pool.append(s)
                        if len(pool) >= n:
                            return pool
    return pool


def _make_sample(pair: tuple[str, str], category: int, frame: str,
                 rng: random.Random) -> dict | None:
    """用给定帧造一条样本；帧与该词对**撞车**时返回 None（换帧重试，由调用方负责）。

    撞车有两种：
      - 帧本身含有待插入的词（`老师让我们区分{w}和{w2}。` 插 `老师` 时该词出现两次）
      - 句子里出现另一个未标注的完整词对
    """
    a, b = pair
    if rng.random() < 0.5:
        a, b = b, a
    text = frame.format(w=a, w2=b)
    if rng.random() < 0.65:
        text = rng.choice(CONTEXT_PREFIXES) + text
    if rng.random() < 0.65:
        text = text.rstrip("。") + rng.choice(CONTEXT_SUFFIXES)
    spans = locate_pair(text, a, b)
    if spans is None:
        return None
    # 句子里是否另有未标注的完整词对？有就丢弃 —— 这条监督信号是错的
    if find_unannotated_pair(text, allowed=(a, b)) is not None:
        return None
    for s in spans:
        s["label"] = category
        s["pair_key"] = f"{a}|{b}"
    return {"text": text, "spans": spans, "category": category, "pair_key": f"{pair[0]}|{pair[1]}"}


def _try_frames(pair: tuple[str, str], category: int, frames: list[str],
                start: int, rng: random.Random) -> tuple[dict | None, int]:
    """按顺序换帧，直到造出一条可用样本。返回 (样本, 下一个帧游标)。

    为什么要换帧而不是直接跳过：帧与词对撞车是**帧×词对**的组合问题，
    一个词对只是恰好碰上了含它自己的那个帧。直接跳过会让这个词对
    在整个数据集里一条样本都没有（实测踩到过 `先生~老师` / `认识~陌生`）。
    """
    for k in range(len(frames)):
        idx = (start + k) % len(frames)
        sample = _make_sample(pair, category, frames[idx], rng)
        if sample is not None:
            return sample, idx + 1
    return None, start


def build_relation_dataset(target_samples: int = 9000, per_pair_floor: int = 8,
                           per_pair_cap: int = 24, bg_ratio: float = 0.30,
                           max_pairs: int | None = None, seed: int = 20240927,
                           corpus_glob: str = CORPUS_GLOB,
                           max_corpus_lines: int | None = None) -> list[dict]:
    """构建近义/反义成对标注数据集。

    Args:
        target_samples: 目标总样本数（背景 / 近义 / 反义三类各自补足到 target_samples//3）
        per_pair_floor: 每个词对至少合成多少条（保证无死角覆盖）
        per_pair_cap: 每个词对最多合成多少条（压平长尾，把预算留给生僻对）
        bg_ratio: 背景句配额比例（下界，实际按 target_samples // 3 取大）
        max_pairs: 每类最多用多少对词（调试用）
    """
    audit = audit_templates()
    if any(w is not None for w in audit["frames"].values()) or audit["background"]:
        print("  ⚠ 模板体检有告警（这些帧在部分词对上会被逐样本丢弃）：")
        for frame, bad in list(audit["frames"].items())[:5]:
            print(f"      帧撞出未标注词对 {bad}: {frame}")
        for s, bad in audit["background"][:5]:
            print(f"      背景句撞出未标注词对 {bad}: {s}")

    rng = random.Random(seed)
    target_per_class = max(target_samples // 3, 1)
    n_bg = max(int(target_samples * bg_ratio), target_per_class)

    buckets: dict[int, list[dict]] = {NONE: [], SYNONYM: [], ANTONYM: []}
    used: dict[str, int] = {}
    skipped = {SYNONYM: 0, ANTONYM: 0}

    for category in (SYNONYM, ANTONYM):
        pairs = list(PAIR_TABLES[category])
        if max_pairs:
            pairs = pairs[:max_pairs]
        frames = NEUTRAL_FRAMES + LEAN_FRAMES[category]
        di = 0

        def emit(pair) -> dict | None:
            nonlocal di
            sample, di = _try_frames(pair, category, frames, di, rng)
            return sample

        for pair in pairs:                       # 1. 保底：每对词恰好 per_pair_floor 条
            for _ in range(per_pair_floor):
                sample = emit(pair)
                if sample is None:
                    skipped[category] += 1
                    continue
                buckets[category].append(sample)
                used[f"{pair[0]}|{pair[1]}"] = used.get(f"{pair[0]}|{pair[1]}", 0) + 1

        while len(buckets[category]) < target_per_class:   # 2. 补齐：轮转仍在上限下的词对
            progressed = False
            for pair in pairs:
                if len(buckets[category]) >= target_per_class:
                    break
                key = f"{pair[0]}|{pair[1]}"
                if used.get(key, 0) >= per_pair_cap:
                    continue
                sample = emit(pair)
                if sample is None:
                    skipped[category] += 1
                    continue
                buckets[category].append(sample)
                used[key] = used.get(key, 0) + 1
                progressed = True
            if not progressed:
                print(f"  ⚠ 类别 {category} 的词对全部达到上限 {per_pair_cap}，"
                      f"只补到 {len(buckets[category])}/{target_per_class}")
                break

    # 背景 = 少量手写句 + 真实语料里挖出来的中性句。真实语料是主力：
    # 只有手写句时 bg_fp 测不出真实误报率（实测 20 句池 → 真实文本误报 8%）。
    good_background = [s for s in BACKGROUND_SENTENCES if find_unannotated_pair(s) is None]
    mined = _mine_neutral(corpus_glob, n_bg * 2, max_lines=max_corpus_lines)
    good_background += [s for s in mined if find_unannotated_pair(s) is None]
    if len(good_background) < 50:
        raise ValueError(
            f"可用背景句只有 {len(good_background)} 条（手写 {len(BACKGROUND_SENTENCES)} + 挖到 {len(mined)}），"
            " 背景太少模型会学成永远开火")
    print(f"  背景句池：手写 {len(BACKGROUND_SENTENCES)} 条 + 真实语料 {len(mined)} 条")
    while len(buckets[NONE]) < n_bg:                                  # 3. 背景类
        buckets[NONE].append({"text": rng.choice(good_background), "spans": []})

    dataset: list[dict] = []
    for c in (NONE, SYNONYM, ANTONYM):
        dataset.extend(buckets[c])
    rng.shuffle(dataset)

    covered = len({d["pair_key"] for d in dataset if d.get("pair_key")})
    print(f"  近义词对 {len(SYNONYM_PAIRS)} / 反义词对 {len(ANTONYM_PAIRS)}，"
          f"实际用到 {covered} 对")
    if any(skipped.values()):
        print(f"  丢弃样本（正例重复/重叠/含未标注词对）: "
              f"近义 {skipped[SYNONYM]} / 反义 {skipped[ANTONYM]}")
    print(f"  成对标注数据集构建完成：{len(dataset)} 样本 "
          f"(背景 {len(buckets[NONE])} / 近义 {len(buckets[SYNONYM])} / 反义 {len(buckets[ANTONYM])}) ✅")
    return dataset


def pair_coverage_report(dataset: list[dict] | None = None) -> dict:
    """每对词的样本量分布 —— 验证「无死角覆盖」这一不变量。"""
    from collections import Counter
    if dataset is None:
        dataset = build_relation_dataset()
    cnt: Counter = Counter()
    for item in dataset:
        if item.get("pair_key"):
            cnt[item["pair_key"]] += 1
    declared = {f"{a}|{b}" for a, b in ALL_PAIRS}
    vals = sorted(cnt.values(), reverse=True)
    return {
        "n_pairs_declared": len(declared),
        "n_pairs_covered": len(cnt),
        "missing": sorted(declared - set(cnt))[:20],
        "n_missing": len(declared - set(cnt)),
        "min": vals[-1] if vals else 0,
        "median": vals[len(vals) // 2] if vals else 0,
        "max": vals[0] if vals else 0,
    }


if __name__ == "__main__":
    ds = build_relation_dataset(target_samples=900, per_pair_floor=1, per_pair_cap=2)
    for item in ds[:6]:
        cat = {0: "背景", 1: "近义", 2: "反义"}[item.get("category", 0)]
        print(f"  [{cat}] {item['text']} -> "
              f"{[(s['word'], s['start'], s['end']) for s in item['spans']]}")
    print(pair_coverage_report(ds))
