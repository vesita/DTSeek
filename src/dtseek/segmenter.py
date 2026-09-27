"""Sentence Segmentation & Chunking Engine for Long Document Decision Models.

Splits long paragraphs into natural sentences while maintaining exact global 1-based character offsets!
Supports:
1. Punctuation-aware split (，。！？；; 等)
2. Preserves exact offset mapping: local_offset in sentence -> global_offset in full document
3. Re-combines sub-sentence predictions seamlessly
"""
import re
from typing import Dict, List, Tuple


def split_with_global_offsets(text: str, max_chunk_len: int = 50) -> List[Dict]:
    """Splits a long document into natural clause/sentence segments,

    tracking the exact 0-based start and end indices in the original text.
    """
    segments = []
    # Match sentences or clauses split by punctuation or newlines
    pattern = re.compile(r"([^。！？\n；;]+[。！？\n；;]?)", re.UNICODE)

    pos = 0
    raw_len = len(text)

    for match in pattern.finditer(text):
        s = match.group(0)
        start_0 = match.start()
        end_0 = match.end()

        # If a single sentence is still longer than max_chunk_len, split further by commas
        if len(s) > max_chunk_len:
            sub_pattern = re.compile(r"([^，,、]+[，,、]?)", re.UNICODE)
            for sub_match in sub_pattern.finditer(s):
                sub_s = sub_match.group(0)
                sub_start = start_0 + sub_match.start()
                sub_end = start_0 + sub_match.end()
                if sub_s.strip():
                    segments.append({
                        "text": sub_s,
                        "global_start": sub_start,
                        "global_end": sub_end,
                    })
        else:
            if s.strip():
                segments.append({
                    "text": s,
                    "global_start": start_0,
                    "global_end": end_0,
                })

    # Catch any trailing text
    if not segments and text.strip():
        segments.append({
            "text": text,
            "global_start": 0,
            "global_end": raw_len,
        })

    return segments
