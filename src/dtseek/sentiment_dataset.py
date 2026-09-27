"""构建对话情绪切片数据集（情绪类别 + 触发词精准闭区间）。

情绪体系（4 分类）：
  - 0: 中性 / 客观描述（背景类）
  - 1: 积极 / 喜悦 / 赞赏
  - 2: 愤怒 / 暴躁 / 不满
  - 3: 悲伤 / 沮丧 / 焦虑

v2 修复（本期）：
  v1 只有 61 个情绪词，且全是"标准书面词"。像 **难受**、**破防了**、**糟心**
  这类最常用的口语/网络表达完全不在表内 —— 用户实测 "我感觉有点难受" 被误判为
  积极。v2 把词典扩到 ~190 词，三个类别各 ~60 词，覆盖书面词 / 口语 / 网络用语；
  并补入模板合成，保证同一个情绪词出现在**句首 / 句中 / 句尾**等不同位置，
  让双指针网络学到位置无关的定位能力。

数据源：
  1. 真实对话挖掘：用扩充后的词典从 nanoSeek 语料命中更多真实句；
  2. 模板合成：为词典里的每个词构造多种上下文，补足真实语料覆盖不到的词。
"""
import glob
import random
import re

# ---------------------------------------------------------------------------
# 情绪词典 v2（~190 词）
#   原则：优先 2 字以上词条，避免单字误匹配 ——
#   例如裸 "赞" 会命中"赞助"、裸 "牛" 会命中"牛奶"、裸 "爽" 会命中"凉爽"。
# ---------------------------------------------------------------------------
EMOTION_KEYWORDS = [
    # 类别 1: 积极 / 喜悦 / 赞赏（60 词）
    (1, [
        # 基础情绪
        "开心", "高兴", "快乐", "愉快", "喜悦", "欢喜", "兴奋", "激动", "幸福", "满足",
        "舒心", "畅快", "痛快", "惬意", "舒服", "舒适", "享受", "轻松",
        # 认同 / 喜爱
        # 注意：**不含"支持"** —— 它在技术语境里是"支持某功能"（能力描述），
        # 不是情绪；收进词典会把"请问支持批量导入吗"这类中性句污染成积极。
        "喜欢", "喜爱", "满意", "赞成", "赞同", "认可", "欣赏", "佩服", "钦佩",
        "羡慕", "期待", "盼望", "惊喜", "惊艳", "赞叹",
        # 评价 / 赞美
        "太棒了", "太好了", "真好", "真棒", "棒极了", "优秀", "出色", "厉害", "给力",
        "靠谱", "点赞", "干得漂亮", "完美", "绝了", "一流", "不错", "挺好", "好极了",
        # 口语 / 网络
        "真香", "爱了", "好评", "巴适", "得劲", "美滋滋", "赞不绝口",
        # 宽慰
        "欣慰", "庆幸", "安心", "踏实", "放心",
    ]),
    # 类别 2: 愤怒 / 暴躁 / 不满（62 词）
    (2, [
        # 愤怒
        "生气", "愤怒", "恼火", "火大", "来气", "气人", "气死", "气炸", "气不过",
        "窝火", "发火", "发脾气", "暴怒", "震怒", "气愤", "愤慨", "暴躁", "急躁",
        "恼羞成怒", "火冒三丈", "大发雷霆",
        # 厌烦
        "烦人", "烦死", "厌烦", "讨厌", "恶心", "反感", "不爽", "憋气", "闹心",
        "糟心", "头大", "不耐烦", "心烦",
        # 差评 / 抱怨
        "差劲", "垃圾", "太烂", "烂透", "坑人", "坑爹", "离谱", "过分", "恶劣",
        "糟糕", "敷衍", "糊弄", "没用", "无用", "白费", "抱怨", "糟透了",
        # 无语 / 斥责
        "无语", "扯淡", "胡扯", "什么鬼", "搞什么", "莫名其妙", "不可理喻",
        "忍无可忍", "受够了", "受不了",
    ]),
    # 类别 3: 悲伤 / 沮丧 / 焦虑（64 词）
    (3, [
        # 悲伤（"难受" 在 v1 里缺失，是本次误判的直接原因）
        "难过", "难受", "伤心", "悲伤", "悲痛", "心痛", "心酸", "委屈", "憋屈",
        "痛苦", "苦闷", "郁闷", "低落", "消沉", "沮丧", "失落", "失望", "绝望",
        "崩溃", "心灰意冷", "心塞", "堵得慌", "揪心", "惆怅", "想哭", "泪目",
        # 焦虑 / 担忧
        "焦虑", "担心", "担忧", "忧虑", "不安", "忐忑", "心慌", "紧张", "害怕",
        "恐惧", "慌张", "发愁", "焦躁", "惶恐", "压力大", "压力好大",
        # 疲惫 / 无力
        "心累", "疲惫", "疲倦", "精疲力尽", "撑不住", "扛不住", "无力", "无助",
        "孤独", "寂寞", "空虚",
        # 口语 / 网络
        "破防了", "绷不住", "麻了", "烦闷", "迷茫", "无奈", "遗憾", "可惜", "舍不得",
    ]),
]


def _validate_lexicon():
    """同一词条不得跨类别（否则切片归属不确定）。"""
    seen = {}
    for cat, words in EMOTION_KEYWORDS:
        for w in words:
            if w in seen:
                raise ValueError(f"情绪词 '{w}' 同时出现在类别 {seen[w]} 与 {cat}")
            seen[w] = cat
    return seen


LEXICON_INDEX = _validate_lexicon()
LEXICON_BY_CAT = {cat: list(words) for cat, words in EMOTION_KEYWORDS}


def _validate_neutral_purity():
    """中性句池**不得含任何情绪词** —— 否则背景类标签被污染。

    这是 fail-closed 断言而不是静默过滤：一旦有人往中性池里加了含情绪词的句子，
    构建数据集时立刻报错，而不是悄悄训出一个"把中性句判成积极"的模型。
    历史上真实踩过：'请问支持批量导入吗' 里的"支持"被当成积极情绪。
    """
    offenders = []
    for s in NEUTRAL_SENTENCES + NEUTRAL_COLLOQUIAL:
        cat, spans = extract_emotion_spans(s)
        if cat != 0:
            offenders.append((s, [x["word"] for x in spans]))
    if offenders:
        detail = "\n".join(f"    '{s}' 命中 {ws}" for s, ws in offenders)
        raise ValueError(
            "中性句池混入情绪词，会造成背景类标签污染：\n" + detail
        )


# ---------------------------------------------------------------------------
# 中性句池：客观陈述句 + 口语中性句（两类都要，否则中庸疑问句会被误判为有情绪）
# ---------------------------------------------------------------------------
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
    "气象台发布大风蓝色预警，请有关单位注意防范。",
    "该方案经过三轮评审，最终确定了实施路径。",
    "缓存命中率提升后，接口平均耗时下降了四成。",
    "新版固件修复了充电协议兼容性问题。",
    "园区绿化改造工程预计在下月完工。",
    "参考文献列出了近五年该领域的主要进展。",
]

NEUTRAL_COLLOQUIAL = [
    "你好，你觉得你现在状态怎么样",
    "请问这个功能具体要怎么使用",
    "我想了解一下完整的操作流程",
    "这个问题大概需要多久能解决",
    "能帮我看看这段代码哪里有问题吗",
    "今天的评审会议几点开始",
    "这个接口的返回格式是什么样的",
    "麻烦确认一下文档里写的参数含义",
    "刚才那个方案是几点发出来的",
    "我需要准备哪些材料才能提交申请",
    "这个版本和上一版有什么区别",
    "请问支持批量导入吗",
    "现在的进度到哪一步了",
    "你那边能不能看到日志输出",
    "这个报错信息对应的原因是什么",
    "稍后把测试结果发我看一下",
    "我们按原计划推进可以吗",
    "这个字段是可选的还是必填的",
    "配置文件放在哪个目录下",
    "麻烦同步一下最新的排期",
    "这部分逻辑是谁负责维护的",
    "现在开始构建会不会影响线上",
    "帮我把这条记录再核对一遍",
    "接下来的步骤分别是什么",
    "这次变更需要走审批流程吗",
    "刚才提到的那份资料在哪里",
    "这个功能预计什么时候上线",
    "方便把复现步骤描述一下吗",
    "环境变量需要配置哪几个",
    "后台任务的执行频率是多少",
]


# ---------------------------------------------------------------------------
# 模板合成：让同一个情绪词出现在句首 / 句中 / 句尾，训练位置无关的双指针定位
# ---------------------------------------------------------------------------
SYNTH_TEMPLATES = [
    "{e}。",                                  # 词在句首
    "{e}，一时不知道怎么形容。",
    "{e}，先这样吧。",
    "说实话，{e}。",                           # 词在句中
    "现在就是{e}。",
    "刚看到这个消息，{e}。",
    "折腾了一整天，{e}。",
    "这波下来，{e}。",
    "想到后面还要继续，{e}。",
    "主要是{e}。",
    "说到底还是{e}。",                          # 词在句尾
    "说不上来，就是{e}。",
    "没办法，{e}。",
    "反正就是{e}。",
    "跟朋友聊完之后，反而{e}。",
]


def _is_negated(text: str, start: int, window: int = 3) -> bool:
    """判断情绪词前方 window 个字符内是否有否定词。

    真实踩过：'我不喜欢你' 会被 "喜欢" 命中标成积极。否定式一律**丢弃**
    （而不是翻转标签）—— "不喜欢" 该算愤怒还是悲伤并不确定，
    丢掉比标错干净。
    """
    prefix = text[max(0, start - window):start]
    return any(neg in prefix for neg in ("不", "没", "别", "未", "无", "非"))


def extract_emotion_spans(text: str) -> tuple[int, list[dict]]:
    """提取句子中的情绪词切片与主导情绪类别。

    Returns:
        (dominant_label, spans)
        dominant_label: 1/2/3 = 主情绪；0 = 无情绪词；-1 = 丢弃（混杂 / 否定式）
    """
    spans = []
    occupied = [False] * len(text)

    # 长词优先，避免 "难受" 被更短的词切开
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
                # 否定式（"不喜欢"）不参与标注，整句丢弃
                if _is_negated(text, idx):
                    return -1, []
                for i in range(idx, end):
                    occupied[i] = True
                spans.append({"label": cat_id, "word": word, "start": idx, "end": end})
            start = idx + 1

    spans.sort(key=lambda x: x["start"])
    if not spans:
        return 0, []

    categories = [s["label"] for s in spans]
    # 积极与愤怒同时出现 → 语义矛盾，丢弃以免污染标签
    if 1 in categories and 2 in categories:
        return -1, []

    return categories[0], spans


def _mine_real(buckets: dict[int, list[dict]], target_per_class: int, max_seq_len: int):
    """用扩充后的词典从真实对话语料挖掘情绪句。"""
    files = sorted(glob.glob("/home/vesita/coding/my/nanoSeek/data/chinese/*dialogue.txt"))
    for f in files:
        if all(len(buckets[c]) >= target_per_class for c in (1, 2, 3)):
            break
        with open(f, encoding="utf-8", errors="ignore") as fp:
            for line in fp:
                if all(len(buckets[c]) >= target_per_class for c in (1, 2, 3)):
                    break
                text = re.sub(r"^(用户|模型|系统|提问|回答|User|Assistant)[:：]\s*", "", line.strip())
                for s in re.split(r"[。！？\n；;]+", text):
                    s = s.strip()
                    if not (4 <= len(s) <= max_seq_len):
                        continue
                    cat_id, spans = extract_emotion_spans(s)
                    if cat_id in (1, 2, 3) and len(buckets[cat_id]) < target_per_class:
                        buckets[cat_id].append({"text": s, "label": cat_id, "spans": spans})


def _synthesize(buckets: dict[int, list[dict]], target_per_class: int, rng: random.Random):
    """真实语料不够时用模板合成补齐（保证每个词都有多种上下文）。"""
    for cat in (1, 2, 3):
        words = LEXICON_BY_CAT[cat]
        guard = 0
        while len(buckets[cat]) < target_per_class and guard < target_per_class * 20:
            guard += 1
            word = rng.choice(words)
            tpl = rng.choice(SYNTH_TEMPLATES)
            s = tpl.format(e=word)
            cat_id, spans = extract_emotion_spans(s)
            # 合成句可能因为模板词与其他词冲突导致类别不符，直接跳过
            if cat_id == cat and spans:
                buckets[cat].append({"text": s, "label": cat, "spans": spans})


def build_sentiment_dataset(target_samples: int = 12000, max_seq_len: int = 64) -> list[dict]:
    """构建情绪切片数据集：真实挖掘优先，模板合成补齐，背景类等量配平。"""
    # fail-closed：中性池被污染时立刻报错，而不是训出一个误判模型
    _validate_neutral_purity()

    rng = random.Random(20240927)
    target_per_class = target_samples // 4

    buckets: dict[int, list[dict]] = {0: [], 1: [], 2: [], 3: []}

    print(f"  情绪数据集目标：每类 {target_per_class} 条（词典 {len(LEXICON_INDEX)} 词）")
    _mine_real(buckets, target_per_class, max_seq_len)
    mined = {c: len(buckets[c]) for c in (1, 2, 3)}
    print(f"  真实语料挖掘：积极 {mined[1]} | 愤怒 {mined[2]} | 悲伤 {mined[3]}")

    _synthesize(buckets, target_per_class, rng)
    synth = {c: len(buckets[c]) - mined[c] for c in (1, 2, 3)}
    print(f"  模板合成补齐：积极 {synth[1]} | 愤怒 {synth[2]} | 悲伤 {synth[3]}")

    # 中性背景类：客观陈述句 + 口语中性句各半
    neutral_pool = NEUTRAL_SENTENCES + NEUTRAL_COLLOQUIAL
    while len(buckets[0]) < target_per_class:
        buckets[0].append({"text": rng.choice(neutral_pool), "label": 0, "spans": []})

    dataset: list[dict] = []
    for c in (0, 1, 2, 3):
        dataset.extend(buckets[c])
    rng.shuffle(dataset)

    print(f"  情绪切片数据集构建完成：总计 {len(dataset)} 样本 ✅")
    for c, name in [(0, "中性/背景"), (1, "积极/喜悦"), (2, "愤怒/不满"), (3, "悲伤/焦虑")]:
        print(f"    类别 {c} ({name}): {len(buckets[c])} 条")
    return dataset


if __name__ == "__main__":
    ds = build_sentiment_dataset(target_samples=2000)
    print()
    for x in ds[:6]:
        print(f"[{x['label']}] '{x['text']}' -> {[(s['word'], s['start'], s['end']) for s in x['spans']]}")
