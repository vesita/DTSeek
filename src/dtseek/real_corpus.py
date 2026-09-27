"""Dataset Pipeline: Harvest and transform nanoSeek real corpora into DTSeek decision samples.

Transforms natural dialogue utterances into:
  - Class 1 (第一人称): 我, 我们, 咱们, 俺, 鄙人, 在下, etc.
  - Class 2 (第二人称): 你, 你们, 您, 阁下, etc.
  - Class 3 (第三人称): 他, 她, 它, 他们, 她们, 它们, etc.
  - Class 0 (无代词/背景类): Sentences completely devoid of any 1st/2nd/3rd person pronouns.
"""
import glob
import os
import random
import re
from typing import Dict, List, Tuple

# Exact pronoun groups
PRONOUNS_1ST = {"我", "我们", "咱们", "俺", "鄙人", "在下", "咱"}
PRONOUNS_2ND = {"你", "你们", "您", "阁下"}
PRONOUNS_3RD = {"他", "她", "它", "他们", "她们", "它们"}

ALL_PRONOUNS = PRONOUNS_1ST | PRONOUNS_2ND | PRONOUNS_3RD


def split_sentences(text: str) -> List[str]:
    """Splits multi-turn dialogues into clean individual sentences."""
    # Remove role tags like '用户：' or '模型：'
    text = re.sub(r"^(用户|模型|系统|提问|回答|User|Assistant)[:：]\s*", "", text.strip())
    # Split by standard sentence punctuation
    parts = re.split(r"[。！？\n；;]+", text)
    cleaned = []
    for p in parts:
        s = p.strip()
        # Keep realistic sentences with 4 to 80 characters
        if 4 <= len(s) <= 80:
            cleaned.append(s)
    return cleaned


def label_sentence(s: str) -> int:
    """Labels a sentence: 1: 1st, 2: 2nd, 3: 3rd, 0: None.
    
    If multiple pronouns exist, label according to the first dominant pronoun,
    or keep pure single-pronoun sentences for unambiguous classification.
    """
    has_1 = any(p in s for p in PRONOUNS_1ST)
    has_2 = any(p in s for p in PRONOUNS_2ND)
    has_3 = any(p in s for p in PRONOUNS_3RD)

    # Clean unambiguous single pronoun class
    if has_1 and not has_2 and not has_3:
        return 1
    if has_2 and not has_1 and not has_3:
        return 2
    if has_3 and not has_1 and not has_2:
        return 3
    if not has_1 and not has_2 and not has_3:
        return 0

    return -1  # Skip mixed pronoun sentences to maintain high data purity


def build_real_pronoun_dataset(
    target_per_class: int = 10000,
    seed: int = 42,
) -> List[Dict]:
    """Extracts, filters, and balances real Chinese sentences directly from nanoSeek dialogue corpora."""
    random.seed(seed)
    corpora_files = sorted(glob.glob("/home/vesita/coding/my/nanoSeek/data/chinese/*dialogue.txt"))
    if not corpora_files:
        raise FileNotFoundError("nanoSeek corpora not found!")

    buckets: Dict[int, List[str]] = {0: [], 1: [], 2: [], 3: []}
    collected_total = 0

    print("Harvesting real Chinese sentences from nanoSeek corpora...")
    for f in corpora_files:
        if all(len(b) >= target_per_class for b in buckets.values()):
            break
        print(f"  scanning {os.path.basename(f)}...")
        with open(f, encoding="utf-8", errors="ignore") as fp:
            for line in fp:
                sentences = split_sentences(line)
                for s in sentences:
                    lbl = label_sentence(s)
                    if lbl in buckets and len(buckets[lbl]) < target_per_class:
                        buckets[lbl].append(s)
                        collected_total += 1
                        if all(len(b) >= target_per_class for b in buckets.values()):
                            break

    print(f"\nHarvested real sentences per category:")
    for k in [0, 1, 2, 3]:
        print(f"  Class {k}: {len(buckets[k])} sentences")

    # Combine and shuffle
    dataset = []
    for k, s_list in buckets.items():
        for s in s_list:
            dataset.append({"text": s, "label": k})

    random.shuffle(dataset)
    print(f"Total balanced dataset size: {len(dataset)} samples ✅\n")
    return dataset


if __name__ == "__main__":
    ds = build_real_pronoun_dataset(target_per_class=100)
    print("Sample items:")
    for x in ds[:8]:
        print(f"  [{x['label']}] {x['text']}")
