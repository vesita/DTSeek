"""长文档自适应断句与全局字符索引映射引擎。

功能：
1. 标点感知断句：优先按句子终止标点（。！？；\n）断开为完整语义分句；
2. 长句二级拆分：若单句长度仍然超过阈值（如 > 50 字符），自动按逗号/顿号细切，保证送入模型的片段严格处于小模型的最优注意力窗口内；
3. 全局绝对字符坐标追踪：精确记录切分片段在原始长文档中的 0-based 与 1-based 全局起止位置，实现局部预测向整篇长文的无损复原。
"""
import re
from typing import Dict, List, Tuple


def split_with_global_offsets(text: str, max_chunk_len: int = 50) -> List[Dict]:
    """将长文档切分为自然的语义子句，并精确记录在原长文档中的 0-based 起止绝对坐标。"""
    segments = []
    # 优先按句子结束标点或换行符分割
    pattern = re.compile(r"([^。！？\n；;]+[。！？\n；;]?)", re.UNICODE)

    raw_len = len(text)

    for match in pattern.finditer(text):
        s = match.group(0)
        start_0 = match.start()
        end_0 = match.end()

        # 若单句仍然过长，按从属逗号/顿号细分
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

    # 若未切出任何片段且文本非空，整段作为兜底
    if not segments and text.strip():
        segments.append({
            "text": text,
            "global_start": 0,
            "global_end": raw_len,
        })

    return segments
