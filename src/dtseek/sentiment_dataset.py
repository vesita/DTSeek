"""构建对话情绪切片分析数据集（包含情绪类别 + 触发词精准闭区间）。

情绪体系 (4 分类)：
  - 0: 中性 / 客观描述 (无明显情绪，背景类)
  - 1: 积极 / 喜悦 / 赞赏 (开心、太棒了、喜欢、舒服、给力...)
  - 2: 愤怒 / 暴躁 / 不满 (烦死了、气人、垃圾、差劲、坑人...)
  - 3: 悲伤 / 沮丧 / 焦虑 (难过、失落、伤心、焦虑、崩溃、绝望...)

数据源：
1. 真实对话挖掘：从 nanoSeek 的 dailychat_dialogue / muice_dialogue / shareai 抽取带有情绪色彩的句子；
2. 经典情绪词汇锚定与跨类别负样本过滤：严格标注起止区间 [start, end]。
"""
import glob
import os
import random
import re
from typing import Dict, List, Tuple

# 情绪触发关键词字典 (按长度降序保证长词优先匹配)
EMOTION_KEYWORDS = [
    # 类别 1: 积极 / 喜悦
    (1, [
        "太棒了", "太好了", "超级开心", "特别棒", "非常满意", "干得漂亮", "太给力了", "很有意思",
        "开心", "高兴", "喜欢", "舒服", "给力", "点赞", "满意", "棒极了", "轻松", "欣慰", "幸福", "赞", "佩服"
    ]),
    # 类别 2: 愤怒 / 暴躁
    (2, [
        "烦死了", "太差劲", "真无语", "气死我了", "搞什么鬼", "什么垃圾", "莫名其妙", "极其恶劣",
        "烦人", "讨厌", "生气", "差劲", "垃圾", "恶心", "离谱", "坑人", "暴躁", "愤怒", "火大", "扯淡"
    ]),
    # 类别 3: 悲伤 / 沮丧 / 焦虑
    (3, [
        "好难过", "太沮丧了", "好心痛", "快崩溃了", "压力好大", "焦虑得不行", "心灰意冷", "特别失落",
        "难过", "伤心", "沮丧", "失望", "委屈", "痛苦", "绝望", "崩溃", "焦虑", "无助", "心酸", "发愁"
    ]),
]

# 纯客观中性文本模板 (背景类，严格无情绪词)
NEUTRAL_SENTENCES = [
    "数据库集群写入延迟保持在五毫秒以内。",
    "系统将于今晚十二点进行常规版本发布与灰度观测。",
    "白日依山尽，黄河入海流。",
    "自动驾驶算法利用多传感器融合进行精准避障。",
    "今天天气晴朗，气温约二十二度。",
    "这份文档详细梳理了底层网络通信协议的技术规范。",
    "高铁路网贯通南北，极大缩短了城际通勤时间。",
    "红富士苹果富含维生素C，口感清甜多汁。",
    "深度学习模型正在加速推理计算过程。",
    "工业机器人按预设计划完成零件焊接组装。",
    "晨曦初现，山林间弥漫着淡淡的薄雾。",
    "春风又绿江南岸，明月何时照江山。",
]


def extract_emotion_spans(text: str) -> Tuple[int, List[Dict]]:
    """提取句子中的情绪关键词切片与主导情绪类别。
    
    返回: (dominant_label, spans_list)
    """
    spans = []
    occupied = [False] * len(text)

    # 展开并按词长降序排列
    flat_keywords = []
    for cat_id, words in EMOTION_KEYWORDS:
        for w in words:
            flat_keywords.append((cat_id, w, len(w)))
    flat_keywords.sort(key=lambda x: -x[2])

    for cat_id, word, length in flat_keywords:
        start = 0
        while True:
            idx = text.find(word, start)
            if idx == -1:
                break
            end = idx + length
            if not any(occupied[i] for i in range(idx, end)):
                for i in range(idx, end):
                    occupied[i] = True
                spans.append({
                    "label": cat_id,
                    "word": word,
                    "start": idx,
                    "end": end,
                })
            start = idx + 1

    spans.sort(key=lambda x: x["start"])

    if not spans:
        return 0, []

    # 统计主导情绪
    categories = [s["label"] for s in spans]
    # 如果同时混杂积极和愤怒，丢弃避免歧义
    if 1 in categories and 2 in categories:
        return -1, []

    dominant_cat = categories[0]
    return dominant_cat, spans


def build_sentiment_dataset(target_samples: int = 12000, max_seq_len: int = 64) -> List[Dict]:
    corpora_files = sorted(glob.glob("/home/vesita/coding/my/nanoSeek/data/chinese/*dialogue.txt"))
    dataset = []

    bucket: Dict[int, List[Dict]] = {0: [], 1: [], 2: [], 3: []}
    target_per_class = target_samples // 4

    print("从 nanoSeek 真实对话语料中挖掘情绪切片样本...")
    for f in corpora_files:
        if all(len(b) >= target_per_class for b in [bucket[1], bucket[2], bucket[3]]):
            break
        with open(f, encoding="utf-8", errors="ignore") as fp:
            for line in fp:
                text = re.sub(r"^(用户|模型|系统|提问|回答|User|Assistant)[:：]\s*", "", line.strip())
                for s in re.split(r"[。！？\n；;]+", text):
                    s = s.strip()
                    if 4 <= len(s) <= max_seq_len:
                        cat_id, spans = extract_emotion_spans(s)
                        if cat_id in (1, 2, 3) and len(bucket[cat_id]) < target_per_class:
                            bucket[cat_id].append({
                                "text": s,
                                "label": cat_id,
                                "spans": spans,
                            })

    # 补充充足的客观中性样本 (类别 0)
    while len(bucket[0]) < target_per_class:
        s = random.choice(NEUTRAL_SENTENCES)
        bucket[0].append({
            "text": s,
            "label": 0,
            "spans": [],
        })

    for k in range(4):
        dataset.extend(bucket[k])

    random.shuffle(dataset)

    print(f"情绪切片数据集构建完成：总计 {len(dataset)} 样本 ✅")
    print(f"  类别 0 (中性/背景): {len(bucket[0])} 条")
    print(f"  类别 1 (积极/喜悦): {len(bucket[1])} 条")
    print(f"  类别 2 (愤怒/不满): {len(bucket[2])} 条")
    print(f"  类别 3 (悲伤/焦虑): {len(bucket[3])} 条")
    return dataset


if __name__ == "__main__":
    ds = build_sentiment_dataset(target_samples=20)
    for x in ds[:5]:
        print(f"[{x['label']}] '{x['text']}' -> spans: {[(s['word'], s['start'], s['end']) for s in x['spans']]}")
