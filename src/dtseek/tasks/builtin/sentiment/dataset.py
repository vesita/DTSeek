"""构建对话情绪切片数据集（情绪类别 + 触发词精准闭区间）。

情绪体系（4 分类）：
  - 0: 中性 / 客观描述（背景类）
  - 1: 积极 / 喜悦 / 赞赏
  - 2: 愤怒 / 暴躁 / 不满
  - 3: 悲伤 / 沮丧 / 焦虑

────────────────────────────────────────────────────────────────────────
v3 修复（问题由词级探针 scripts/eval_emotion_words.py 暴露）

v2 的假象：句级验证集准确率 98%，但**词级探针只有 43.7%**（悲伤类甚至 13.3%）。
根因：真实语料挖掘是 Zipf 分布 ——
    类别1 词典 63 词，训练里只出现 41 词，样本量中位数 = 2，22 个词从未出现；
    类别3 词典 62 词，只出现 43 词，中位数 = 3，19 个词从未出现。
高频词（"难受" 1308 条）淹没低频词（"心塞" 0 条），验证集又抽的是同一批高频词，
所以 98% 完全是"在自己会的那批词上考自己"。

v3 三条修复：
  1. **每个词一条下限**（per_word_floor）：合成保底，保证词典里每个词至少有
     N 条训练样本，消灭"从未出现"；
  2. **每个词一条上限**（per_word_cap）：真实挖掘按首切片词计数封顶，
     压平 Zipf 长尾，避免高频词淹没低频词；
  3. **显式否定式**（"不高兴/不开心/不满意"）：v2 把否定句整句丢弃，
     模型从没见过否定式；v3 把常见否定式作为词条收进负类，
     使"高兴→积极"与"不高兴→消极"同时可学。
────────────────────────────────────────────────────────────────────────
"""
import glob
import random
import re

# ---------------------------------------------------------------------------
# 情绪词典 v3（~200 词）
#   原则：优先 2 字以上词条，避免单字误匹配 ——
#   裸 "赞" 会命中"赞助"、裸 "牛" 会命中"牛奶"、裸 "爽" 会命中"凉爽"。
# ---------------------------------------------------------------------------
EMOTION_KEYWORDS = [
    # ── 类别 1: 积极 / 喜悦 / 赞赏 ──
    (1, [
        # 基础情绪（"高兴/开心"这一类最简单直接的核心词）
        "开心", "高兴", "快乐", "愉快", "喜悦", "欢喜", "兴奋", "激动", "幸福", "满足",
        "舒心", "畅快", "痛快", "惬意", "舒服", "舒适", "享受", "轻松",
        # 认同 / 喜爱
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
    # ── 类别 2: 愤怒 / 暴躁 / 不满 ──
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
        # ★ 显式否定式（对"满意/喜欢"的否定属于不满，不是悲伤）
        "不满意", "不喜欢", "不认可", "不赞同", "看不上",
    ]),
    # ── 类别 3: 悲伤 / 沮丧 / 焦虑 ──
    (3, [
        # 悲伤
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
        # ★ 显式否定式（对"高兴/开心"的否定属于情绪低落）
        "不高兴", "不开心", "不快乐", "不愉快", "不舒服", "不痛快", "不踏实", "不放心",
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


# ---------------------------------------------------------------------------
# 中性句池：客观陈述句 + 口语中性句
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
# 合成载体模板：词出现在句首 / 句中 / 句尾，句式各不相同。
#   模板自身不得含情绪词（由 _validate_carriers 强制）。
# ---------------------------------------------------------------------------
SYNTH_CARRIERS = [
    # 词在句首
    "{e}。", "{e}，真的。", "{e}啊。", "{e}，谁懂啊。",
    "{e}，一时不知道怎么形容。", "{e}，先这样吧。",
    # 词在句中
    "说实话，{e}。", "现在就是{e}。", "主要是{e}。", "说到底还是{e}。",
    "刚看到这个消息，{e}。", "折腾了一整天，{e}。", "这波下来，{e}。",
    "想到后面还要继续，{e}。", "跟朋友聊完之后，{e}。",
    "说不上来，就是{e}。", "反正就是{e}。", "没办法，{e}。",
    "这几天一直{e}。", "遇到这种事，{e}。", "看到评论区，{e}。",
    # 词在句尾
    "今天真的是{e}。", "我现在整个人都{e}。", "听完这话我直接{e}。",
    "忙完这些事只剩{e}。", "最近的状态就是{e}。", "想想还是有点{e}。",
    "说完之后反而更{e}。", "不知道为啥就是{e}。", "整个人处于{e}的状态。",
    # 带前后语境的复合句
    "本来没觉得什么，后来越想越{e}。",
    "事情已经这样了，再说什么也是{e}。",
    "他这么一说，我突然觉得{e}。",
    "虽然不该这样，但还是{e}。",
    "过了这么久，提起来依然{e}。",
    "今天发生了很多事，总之{e}。",
    # ★ 无终止标点（真实聊天输入常常没有句号）
    #   缺这一类会让模型在"无标点短输入"上定位截断：
    #   实测 '开心' -> 只圈 '开'，'心情不错' -> 误判愤怒；补上后两者均正常。
    "{e}", "真的很{e}", "我现在{e}", "感觉{e}", "有点{e}",
    "整个人都{e}", "确实是{e}", "怎么这么{e}", "好{e}",
]


def _validate_carriers():
    """载体模板自身不得含情绪词 —— 否则合成句会出现多切片、标签归属混乱。"""
    bad = []
    for t in SYNTH_CARRIERS:
        probe = t.format(e="")
        cat, spans = extract_emotion_spans(probe)
        if cat != 0:
            bad.append((t, [x["word"] for x in spans]))
    if bad:
        raise ValueError("合成载体模板本身含情绪词：\n" +
                         "\n".join(f"    '{t}' 命中 {w}" for t, w in bad))


def _validate_neutral_purity():
    """中性句池不得含任何情绪词（否则背景类标签被污染）。

    fail-closed 断言而非静默过滤：一旦往中性池加了含情绪词的句子，立刻报错。
    真实踩过：'请问支持批量导入吗' 里的"支持"曾被当成积极情绪（该词已从词典移除）。
    """
    offenders = []
    for s in NEUTRAL_SENTENCES + NEUTRAL_COLLOQUIAL:
        cat, spans = extract_emotion_spans(s)
        if cat != 0:
            offenders.append((s, [x["word"] for x in spans]))
    if offenders:
        detail = "\n".join(f"    '{s}' 命中 {ws}" for s, ws in offenders)
        raise ValueError("中性句池混入情绪词，会造成背景类标签污染：\n" + detail)


def _is_negated(text: str, start: int, window: int = 3) -> bool:
    """判断情绪词前方 window 个字符内是否有否定词。

    注意：词典已显式收录"不高兴/不开心/不满意"等常见否定式，它们会作为
    **整体词条**先被匹配到（长词优先），不会走到这里。这里兜底的是词典未收录的
    临时否定；命中则整句丢弃 —— 翻转标签风险更大，丢掉比标错干净。
    """
    prefix = text[max(0, start - window):start]
    return any(neg in prefix for neg in ("不", "没", "别", "未", "无", "非"))


def extract_emotion_spans(text: str):
    """提取句子中的情绪词切片与主导情绪类别。

    Returns:
        (dominant_label, spans)
        dominant_label: 1/2/3 = 主情绪；0 = 无情绪词；-1 = 丢弃（混杂 / 未收录的否定式）
    """
    spans = []
    occupied = [False] * len(text)

    # 长词优先："不高兴" 必须先于 "高兴" 匹配
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


# ---------------------------------------------------------------------------
# 数据集构建：合成保底（每词有下限）+ 真实挖掘（每词有上限）
# ---------------------------------------------------------------------------

def _synthesize_floor(buckets, per_word_floor: int):
    """为词典里**每个词**生成恰好 per_word_floor 条样本，保证无死角覆盖。"""
    for cat in (1, 2, 3):
        words = LEXICON_BY_CAT[cat]
        for wi, word in enumerate(words):
            made = 0
            ti = wi  # 用词序号做起点偏移，避免所有词都从同一个模板开始
            guard = 0
            while made < per_word_floor and guard < per_word_floor * 50:
                guard += 1
                tpl = SYNTH_CARRIERS[ti % len(SYNTH_CARRIERS)]
                ti += 1
                s = tpl.format(e=word)
                cat_id, spans = extract_emotion_spans(s)
                # 合成句必须只标出这个词、且类别正确，否则丢弃重试
                if cat_id == cat and spans and len(spans) == 1:
                    buckets[cat].append({"text": s, "label": cat, "spans": spans})
                    made += 1


def _mine_real_capped(buckets, target_per_class: int, max_seq_len: int,
                      per_word_cap: int):
    """真实语料挖掘，按首切片词计数**封顶**，压平 Zipf 长尾。

    不加封顶时高频词（"难受" 1300+ 条）会淹没低频词（几十条甚至 0 条），
    这正是 v2 词级准确率只有 43.7% 的直接原因。
    """
    files = sorted(glob.glob("/home/vesita/coding/my/nanoSeek/data/chinese/*dialogue.txt"))
    word_count = {c: {} for c in (1, 2, 3)}

    def _full():
        return all(len(buckets[c]) >= target_per_class for c in (1, 2, 3))

    for f in files:
        if _full():
            break
        with open(f, encoding="utf-8", errors="ignore") as fp:
            for line in fp:
                if _full():
                    break
                text = re.sub(r"^(用户|模型|系统|提问|回答|User|Assistant)[:：]\s*", "", line.strip())
                for s in re.split(r"[。！？\n；;]+", text):
                    s = s.strip()
                    if not (4 <= len(s) <= max_seq_len):
                        continue
                    cat_id, spans = extract_emotion_spans(s)
                    if cat_id not in (1, 2, 3):
                        continue
                    if len(buckets[cat_id]) >= target_per_class:
                        continue
                    head_word = spans[0]["word"]
                    cnt = word_count[cat_id].get(head_word, 0)
                    if cnt >= per_word_cap:
                        continue  # 该词已达上限，跳过（给长尾留空间）
                    word_count[cat_id][head_word] = cnt + 1
                    buckets[cat_id].append({"text": s, "label": cat_id, "spans": spans})


def build_sentiment_dataset(target_samples: int = 32000, max_seq_len: int = 64,
                            per_word_floor: int = 60, per_word_cap: int = 120,
                            seed: int = 20240927):
    """构建情绪切片数据集。

    Args:
        target_samples: 目标总样本数（四类均分）
        per_word_floor: 每个词至少合成多少条（保证无死角覆盖）
        per_word_cap: 每个词的真实语料最多采多少条（压平 Zipf 长尾）
        seed: 随机种子；默认值是历史固定值，改动它会改变数据集内容
    """
    _validate_carriers()
    _validate_neutral_purity()

    rng = random.Random(seed)
    target_per_class = target_samples // 4
    buckets = {0: [], 1: [], 2: [], 3: []}

    n_words = {c: len(LEXICON_BY_CAT[c]) for c in (1, 2, 3)}
    print(f"  情绪词典：{len(LEXICON_INDEX)} 词 "
          f"(积极 {n_words[1]} / 愤怒 {n_words[2]} / 悲伤 {n_words[3]})")

    # 1. 合成保底：每个词恰好 per_word_floor 条
    _synthesize_floor(buckets, per_word_floor)
    floor_n = {c: len(buckets[c]) for c in (1, 2, 3)}
    print(f"  合成保底(每词 {per_word_floor} 条)："
          f"积极 {floor_n[1]} / 愤怒 {floor_n[2]} / 悲伤 {floor_n[3]}")

    # 2. 真实语料补齐（单词封顶 per_word_cap）
    _mine_real_capped(buckets, target_per_class, max_seq_len, per_word_cap)
    real_n = {c: len(buckets[c]) - floor_n[c] for c in (1, 2, 3)}
    print(f"  真实语料补充(每词上限 {per_word_cap})："
          f"积极 {real_n[1]} / 愤怒 {real_n[2]} / 悲伤 {real_n[3]}")

    # 3. 中性背景类
    neutral_pool = NEUTRAL_SENTENCES + NEUTRAL_COLLOQUIAL
    while len(buckets[0]) < target_per_class:
        buckets[0].append({"text": rng.choice(neutral_pool), "label": 0, "spans": []})

    dataset = []
    for c in (0, 1, 2, 3):
        dataset.extend(buckets[c])
    rng.shuffle(dataset)

    print(f"  情绪切片数据集构建完成：总计 {len(dataset)} 样本 ✅")
    for c, name in [(0, "中性/背景"), (1, "积极/喜悦"), (2, "愤怒/不满"), (3, "悲伤/焦虑")]:
        print(f"    类别 {c} ({name}): {len(buckets[c])} 条")
    return dataset


def word_coverage_report(dataset=None) -> dict:
    """统计每个词在数据集里的样本量，用于验证"无死角覆盖"这一不变量。"""
    from collections import Counter
    if dataset is None:
        dataset = build_sentiment_dataset()
    per_word = {c: Counter() for c in (1, 2, 3)}
    for item in dataset:
        for s in item.get("spans", []):
            per_word[s["label"]][s["word"]] += 1
    report = {}
    for c in (1, 2, 3):
        vocab = LEXICON_BY_CAT[c]
        cnt = per_word[c]
        vals = sorted((cnt[w] for w in vocab), reverse=True)
        report[c] = {
            "n_words": len(vocab),
            "covered": sum(1 for w in vocab if cnt[w] > 0),
            "zero": [w for w in vocab if cnt[w] == 0],
            "min": min(vals), "median": vals[len(vals) // 2], "max": max(vals),
        }
    return report


if __name__ == "__main__":
    ds = build_sentiment_dataset(target_samples=4000)
    print()
    rep = word_coverage_report(ds)
    for c, r in rep.items():
        print(f"类别 {c}: 覆盖 {r['covered']}/{r['n_words']} 词  "
              f"min={r['min']} 中位={r['median']} max={r['max']}  未覆盖={r['zero']}")
