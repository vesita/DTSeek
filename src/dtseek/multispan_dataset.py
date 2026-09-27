"""Multi-Span Dataset Pipeline from nanoSeek corpora.

A single sentence can contain 0, 1, 2, 3 or more pronoun spans simultaneously:
e.g. "我把方案发给你，他在等我们" contains:
  - "我": [1, 1], Class 1
  - "你": [7, 7], Class 2
  - "他": [9, 9], Class 3
  - "我们": [14, 15], Class 1
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


def extract_all_spans(text: str) -> List[Dict]:
    """Finds all non-overlapping pronoun spans in a sentence."""
    spans = []
    occupied = [False] * len(text)

    # Search longer phrases first to avoid partial splits (e.g. "我们" before "我")
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
            # Check overlap
            if not any(occupied[i] for i in range(idx, end)):
                for i in range(idx, end):
                    occupied[i] = True
                spans.append({
                    "label": cat_id,
                    "word": p,
                    "start": idx,
                    "end": end,
                    "center": (idx + end) / 2.0 / len(text),
                    "width": length / len(text),
                })
            start = idx + 1

    spans.sort(key=lambda x: x["start"])
    return spans


def build_multispan_dataset(target_samples: int = 12000, max_seq_len: int = 64) -> List[Dict]:
    corpora_files = sorted(glob.glob("/home/vesita/coding/my/nanoSeek/data/chinese/*dialogue.txt"))
    dataset = []

    print("Harvesting multi-span real sentences from nanoSeek corpora...")
    for f in corpora_files:
        if len(dataset) >= target_samples:
            break
        with open(f, encoding="utf-8", errors="ignore") as fp:
            for line in fp:
                text = re.sub(r"^(用户|模型|系统|提问|回答|User|Assistant)[:：]\s*", "", line.strip())
                for s in re.split(r"[。！？\n；;]+", text):
                    s = s.strip()
                    if 4 <= len(s) <= max_seq_len:
                        spans = extract_all_spans(s)
                        dataset.append({
                            "text": s,
                            "spans": spans,
                            "num_spans": len(spans),
                        })
                        if len(dataset) >= target_samples:
                            break

    random.shuffle(dataset)
    span_counts = {}
    for d in dataset:
        c = d["num_spans"]
        span_counts[c] = span_counts.get(c, 0) + 1

    print(f"Collected {len(dataset)} multi-span samples ✅")
    print(f"  Distribution of span counts: {sorted(span_counts.items())[:6]}")
    return dataset


if __name__ == "__main__":
    ds = build_multispan_dataset(target_samples=20)
    for x in ds[:5]:
        print(f"Text: '{x['text']}' -> {x['num_spans']} spans: {[(s['word'], s['start'], s['end']) for s in x['spans']]}")
