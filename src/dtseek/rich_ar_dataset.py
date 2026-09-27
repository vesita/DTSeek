"""Multi-Span Dataset Builder with Rich Multi-Pronoun Sentences & Synthetic Augmentations.

Guarantees high representation of sentences with 2, 3, and 4+ pronouns to prevent premature <eos>!
"""
import glob
import os
import random
import re
from typing import Dict, List, Tuple

PRONOUN_MAP = [
    (1, ["我们", "咱们", "鄙人", "在下", "我", "俺", "咱"]),
    (2, ["你们", "阁下", "你", "您"]),
    (3, ["他们", "她们", "它们", "他", "她", "它"]),
]

SYNTHETIC_TEMPLATES = [
    "你好，请问你知道{p1}这句话是什么意思吗？",
    "{p1}刚才和{p2}商量了一下，觉得这个方案非常可行。",
    "如果{p1}有任何疑问，随时向{p2}提出，{p3}也会一起协助解答。",
    "{p1}把代码提交给{p2}审查，随后{p3}在测试环境部署。",
    "大家都在等待，看{p1}和{p2}谁能先完成模块开发。",
    "听说明天下午开会，{p1}和{p2}准备好各自的汇报PPT了吗？",
    "{p1}非常感谢{p2}这段时间的耐心指导，让{p3}受益匪浅。",
    "每次系统发布，{p1}都会提醒{p2}仔细检查监控指标。",
]


def extract_all_spans(text: str) -> List[Dict]:
    spans = []
    occupied = [False] * len(text)

    all_pronouns = []
    for cat_id, p_list in PRONOUN_MAP:
        for p in p_list:
            all_pronouns.append((cat_id, p, len(p)))
    all_pronouns.sort(key=lambda x: -x[2])

    for cat_id, p, length in all_pronouns:
        start = 0
        while True:
            idx = text.find(p, start)
            if idx == -1:
                break
            end = idx + length
            if not any(occupied[i] for i in range(idx, end)):
                for i in range(idx, end):
                    occupied[i] = True
                spans.append({
                    "label": cat_id,
                    "word": p,
                    "start": idx,
                    "end": end,
                })
            start = idx + 1

    spans.sort(key=lambda x: x["start"])
    return spans


def build_rich_ar_dataset(target_samples: int = 15000, max_seq_len: int = 64) -> List[Dict]:
    corpora_files = sorted(glob.glob("/home/vesita/coding/my/nanoSeek/data/chinese/*dialogue.txt"))
    dataset = []

    # 1. Harvest real sentences, prioritizing multi-pronoun cases (num_spans >= 2)
    bucket_multi = []
    bucket_single = []
    bucket_zero = []

    print("Harvesting and categorizing real sentences from nanoSeek corpora...")
    for f in corpora_files:
        if len(bucket_multi) >= target_samples // 2:
            break
        with open(f, encoding="utf-8", errors="ignore") as fp:
            for line in fp:
                text = re.sub(r"^(用户|模型|系统|提问|回答|User|Assistant)[:：]\s*", "", line.strip())
                for s in re.split(r"[。！？\n；;]+", text):
                    s = s.strip()
                    if 4 <= len(s) <= max_seq_len:
                        spans = extract_all_spans(s)
                        item = {"text": s, "spans": spans, "num_spans": len(spans)}
                        if len(spans) >= 2:
                            bucket_multi.append(item)
                        elif len(spans) == 1:
                            if len(bucket_single) < target_samples // 3:
                                bucket_single.append(item)
                        else:
                            if len(bucket_zero) < target_samples // 4:
                                bucket_zero.append(item)

    # 2. Add targeted synthetic multi-pronoun sentences (e.g. "你好...你知道我...")
    p1_candidates = ["我", "我们", "咱们"]
    p2_candidates = ["你", "您", "你们"]
    p3_candidates = ["他", "她", "他们"]

    synthetic_samples = []
    for _ in range(2500):
        p1 = random.choice(p1_candidates)
        p2 = random.choice(p2_candidates)
        p3 = random.choice(p3_candidates)
        tpl = random.choice(SYNTHETIC_TEMPLATES)
        s = tpl.format(p1=p1, p2=p2, p3=p3)
        spans = extract_all_spans(s)
        synthetic_samples.append({"text": s, "spans": spans, "num_spans": len(spans)})

    dataset = bucket_multi + bucket_single + bucket_zero + synthetic_samples
    random.shuffle(dataset)

    counts = {}
    for d in dataset:
        c = d["num_spans"]
        counts[c] = counts.get(c, 0) + 1

    print(f"Total Rich AR Dataset: {len(dataset)} samples ✅")
    print(f"  Distribution of span counts: {sorted(counts.items())[:6]}")
    return dataset
