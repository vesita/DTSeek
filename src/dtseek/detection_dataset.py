"""Dataset Pipeline with YOLO-style Bounding Span Labels (start_idx, end_idx).

For each sentence:
  - Class: 1: 1st, 2: 2nd, 3: 3rd, 0: None
  - Span: Character character start and end index (e.g., [4, 6] for "我们")
  - Normalized Box: (center, width) in [0, 1] relative to sequence length
"""
import glob
import os
import random
import re
from typing import Dict, List, Optional, Tuple

PRONOUNS_1ST = ["我们", "咱们", "鄙人", "在下", "我", "俺", "咱"]
PRONOUNS_2ND = ["你们", "阁下", "你", "您"]
PRONOUNS_3RD = ["他们", "她们", "它们", "他", "她", "它"]


def find_pronoun_span(text: str) -> Tuple[int, int, int, str]:
    """Finds single unambiguous pronoun and its character span (start, end).
    
    Returns:
        (category_id, start_char, end_char, matched_word)
        or (0, 0, 0, "") if no pronoun found.
        or (-1, 0, 0, "") if ambiguous / mixed pronouns.
    """
    found = []
    # Check 1st person
    for p in PRONOUNS_1ST:
        idx = text.find(p)
        if idx != -1:
            found.append((1, idx, idx + len(p), p))
    # Check 2nd person
    for p in PRONOUNS_2ND:
        idx = text.find(p)
        if idx != -1:
            found.append((2, idx, idx + len(p), p))
    # Check 3rd person
    for p in PRONOUNS_3RD:
        idx = text.find(p)
        if idx != -1:
            found.append((3, idx, idx + len(p), p))

    if not found:
        return 0, 0, 0, ""

    categories = set(x[0] for x in found)
    if len(categories) > 1:
        return -1, 0, 0, ""  # Mixed categories, drop to keep clean signal

    # Pick the longest match
    found.sort(key=lambda x: -(x[2] - x[1]))
    cat, start, end, word = found[0]
    return cat, start, end, word


def build_detection_dataset(target_per_class: int = 4000, max_seq_len: int = 64) -> List[Dict]:
    corpora_files = sorted(glob.glob("/home/vesita/coding/my/nanoSeek/data/chinese/*dialogue.txt"))
    buckets: Dict[int, List[Dict]] = {0: [], 1: [], 2: [], 3: []}

    print("Harvesting YOLO-style detection spans from nanoSeek corpora...")
    for f in corpora_files:
        if all(len(b) >= target_per_class for b in buckets.values()):
            break
        with open(f, encoding="utf-8", errors="ignore") as fp:
            for line in fp:
                text = re.sub(r"^(用户|模型|系统|提问|回答|User|Assistant)[:：]\s*", "", line.strip())
                for s in re.split(r"[。！？\n；;]+", text):
                    s = s.strip()
                    if 4 <= len(s) <= max_seq_len:
                        cat, start, end, word = find_pronoun_span(s)
                        if cat in buckets and len(buckets[cat]) < target_per_class:
                            L = len(s)
                            if cat == 0:
                                norm_center = 0.0
                                norm_width = 0.0
                            else:
                                norm_center = (start + end) / 2.0 / L
                                norm_width = (end - start) / L

                            buckets[cat].append({
                                "text": s,
                                "label": cat,
                                "start": start,
                                "end": end,
                                "word": word,
                                "center": norm_center,
                                "width": norm_width,
                            })
                            if all(len(b) >= target_per_class for b in buckets.values()):
                                break

    dataset = []
    for k, v in buckets.items():
        dataset.extend(v)
    random.shuffle(dataset)
    print(f"Constructed detection dataset: {len(dataset)} samples (balanced across 4 classes) ✅")
    return dataset


if __name__ == "__main__":
    ds = build_detection_dataset(target_per_class=10)
    for x in ds[:5]:
        print(f"[{x['label']}] '{x['text']}' -> span: [{x['start']}:{x['end']}] ('{x['word']}') box: (c={x['center']:.3f}, w={x['width']:.3f})")
