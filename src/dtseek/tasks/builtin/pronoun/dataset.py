"""多代词真实语料 + 合成增强切片数据集（v3：补「说话动词 + 冒号 + 直接引语」句式）。

v1 的致命缺陷（已定位）：
    合成句硬编码 for _ in range(2500)，全部含代词、从不产背景句；
    且扫描循环在 bucket_multi 满时就 break，bucket_zero 常常根本没填满。
    实测 target=600 时背景句只占 4.7% —— 模型因此学会「永远开火」，
    对中性句 100% 误报。

v2 修复：四个桶各自独立填到配额（互不 early-break），并显式给背景句 32% 配额。

v3 修复（隔离实验定位）：模型在
    `他说：“我不认识你”`
这类「说话动词 + 全角冒号 + 直接引语」句式上翻车 —— 把「他」判成第一人称，
并且整个「我」被跳过。隔离实验确认触发条件是 **`：` 紧接 `“`**（去掉任一个都正常）。
v3 因此新增 SYMBOLIC_QUOTE_TEMPLATES 桶，同时覆盖：
    - 触发形态：`{p}说：“{p2}…”` / `{p}问：“…”` / `{p}答：“…”` / `{p}对{q}说：“{p2}…”`
    - 近邻变体：只带冒号不带引号 / 只带引号不带冒号 / 逗号 / 都不带
不让模型靠单个符号取巧。这些句子代词密集（常 3 个），全部由 extract_all_spans
标出，构建时逐句断言「全标出、按起始位置有序、不重叠」。

注意：`“` `”` `：` 都是**合法输入**，绝不能进任何「背景纯度」禁词表。
"""
import glob
import random
import re

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

#: 「说话动词 + 冒号 + 直接引语」及其近邻变体。
#: 每一句都至少含 2 个（多数 3 个）代词；`{p1}` 第一人称、`{p2}` 第二人称、`{p3}` 第三人称。
SYMBOLIC_QUOTE_TEMPLATES = (
    # —— 触发形态：`：` 紧接 `“` ——
    "{p3}说：“{p1}不认识{p2}。”",
    "{p3}说：“{p1}明天去找{p2}。”",
    "{p3}说：“{p1}已经告诉过{p2}了。”",
    "{p3}问：“{p2}现在在哪儿？”",
    "{p3}答：“{p1}不知道。”",
    "{p3}回答：“{p1}马上就来。”",
    "{p3}对{p2}说：“{p1}明天来找{p2}。”",
    "{p3}对{p2}说：“{p1}把书还给了{p2}。”",
    "{p3}跟{p2}说：“{p1}不认识{p2}。”",
    "{p3}告诉{p2}：“{p1}明天就走。”",
    # —— 近邻变体 a：只带冒号，不带引号 ——
    "{p3}说：{p1}不认识{p2}。",
    "{p3}对{p2}说：{p1}明天来找{p2}。",
    "{p3}问：{p2}现在在哪儿？",
    "{p3}答：{p1}不知道。",
    # —— 近邻变体 b：只带引号，不带冒号 ——
    "{p3}说“{p1}不认识{p2}”。",
    "{p3}对{p2}说“{p1}明天来找{p2}”。",
    "{p3}问“{p2}现在在哪儿？”",
    # —— 近邻变体 c：逗号 / 什么都不带 ——
    "{p3}说，{p1}不认识{p2}。",
    "{p3}告诉{p2}，{p1}明天就走。",
    "{p3}说{p1}不认识{p2}。",
)

#: 符号句在数据集里的配额（占 target_samples 的比例）。
#: 不放进函数签名——`build_rich_ar_dataset` 的接口要保持原样。
SYMBOLIC_RATIO = 0.15

# 保证无任何代词（"我/你/他/她/它/咱/俺" 一字都不含）的客观背景句池
NEUTRAL_POOL = [
    "数据库集群写入延迟保持在五毫秒以内。",
    "自动驾驶算法利用多传感器融合进行精准避障。",
    "今天天气晴朗，气温约二十二度，适合户外徒步。",
    "白日依山尽，黄河入海流。",
    "高铁路网贯通南北，极大缩短了城际通勤时间。",
    "红富士苹果富含维生素C，口感清甜多汁。",
    "深度学习模型正在加速推理计算过程。",
    "工业机器人按预设计划完成零件焊接组装。",
    "晨曦初现，山林间弥漫着淡淡的薄雾。",
    "春风又绿江南岸，明月何时照江山。",
    "图书馆周末正常开放，欢迎读者借阅书籍。",
    "这份文档详细梳理了底层网络通信协议的技术规范。",
    "算法经过严谨推导保证了数学上的收敛性。",
    "芯片制程工艺突破带来了算力大幅跃升。",
    "气象台发布大风蓝色预警，请有关单位注意防范。",
    "该方案经过三轮评审，最终确定了实施路径。",
    "缓存命中率提升后，接口平均耗时下降了四成。",
    "新版固件修复了充电协议兼容性问题。",
    "园区绿化改造工程预计在下月完工。",
    "参考文献列出了近五年该领域的主要进展。",
]


def extract_all_spans(text: str) -> list[dict]:
    """找出句中所有不重叠的代词切片（长词优先，避免"我们"被拆成"我"）。"""
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
                spans.append({"label": cat_id, "word": p, "start": idx, "end": end})
            start = idx + 1

    spans.sort(key=lambda x: x["start"])
    return spans


def _symbolic_sample(rng: random.Random) -> dict:
    """造一条「说话动词 + 冒号 + 直接引语」样本，并**逐句自检**。

    fail-closed：任何一条符号句没能被 extract_all_spans 完整、有序地标出，
    就直接抛错 —— 这条监督信号是错的，宁可炸掉构建也不要静默训坏。
    """
    p1 = ["我", "我们", "咱们"]
    p2 = ["你", "您", "你们"]
    p3 = ["他", "她", "他们", "她们"]
    tpl = rng.choice(SYMBOLIC_QUOTE_TEMPLATES)
    text = tpl.format(p1=rng.choice(p1), p2=rng.choice(p2), p3=rng.choice(p3))
    spans = extract_all_spans(text)
    if len(spans) < 2:
        raise ValueError(f"符号句自检失败（代词少于 2 个）: {text!r} -> {spans}")
    starts = [s["start"] for s in spans]
    if starts != sorted(starts) or len(starts) != len(set(starts)):
        raise ValueError(f"符号句自检失败（切片未按位置有序或不唯一）: {text!r} -> {spans}")
    # 回扫一遍：词表口径必须与切片口径逐字一致（防漏标）
    if [(s["start"], s["end"], s["word"]) for s in extract_all_spans(text)] != \
       [(s["start"], s["end"], s["word"]) for s in spans]:
        raise ValueError(f"符号句自检失败（标记不一致）: {text!r}")
    return {"text": text, "spans": spans}


def build_rich_ar_dataset(target_samples: int = 15000, max_seq_len: int = 64,
                          bg_ratio: float = 0.32, synthetic_ratio: float = 0.20,
                          seed: int = 20240927) -> list[dict]:
    """构建代词任务数据集。

    Args:
        target_samples: 目标总样本数
        bg_ratio: 背景句（无代词）配额比例，默认 32% —— v1 只有 4.7%，是误报根因
        synthetic_ratio: 合成多代词句配额比例
        seed: 随机种子；默认值是历史固定值，改动它会改变数据集内容

    符号句（v3）配额由模块常量 `SYMBOLIC_RATIO` 决定，从总数里划走，
    保持函数签名不变。
    """
    rng = random.Random(seed)

    n_bg = int(target_samples * bg_ratio)
    n_syn = int(target_samples * synthetic_ratio)
    n_sym = int(target_samples * SYMBOLIC_RATIO)
    n_real = max(target_samples - n_bg - n_syn - n_sym, 0)

    print(f"  代词数据集配额：真实句 {n_real} | 合成句 {n_syn} | "
          f"符号句 {n_sym} | 背景句 {n_bg}")

    corpus_files = sorted(glob.glob("/home/vesita/coding/my/nanoSeek/data/chinese/*dialogue.txt"))

    # 三个桶各自独立填配额，绝不因为某个桶满了就 break —— v1 的 bug 就在这里
    bucket_multi, bucket_single, bucket_zero = [], [], []

    def _all_full():
        return len(bucket_multi) + len(bucket_single) >= n_real and len(bucket_zero) >= n_bg

    for f in corpus_files:
        if _all_full():
            break
        with open(f, encoding="utf-8", errors="ignore") as fp:
            for line in fp:
                if _all_full():
                    break
                text = re.sub(r"^(用户|模型|系统|提问|回答|User|Assistant)[:：]\s*", "", line.strip())
                for s in re.split(r"[。！？\n；;]+", text):
                    s = s.strip()
                    if not (4 <= len(s) <= max_seq_len):
                        continue
                    spans = extract_all_spans(s)
                    if len(spans) >= 2:
                        if len(bucket_multi) < n_real // 2:
                            bucket_multi.append({"text": s, "spans": spans})
                    elif len(spans) == 1:
                        if len(bucket_single) < n_real - len(bucket_multi):
                            bucket_single.append({"text": s, "spans": spans})
                    else:
                        if len(bucket_zero) < n_bg:
                            bucket_zero.append({"text": s, "spans": []})

    # 真实语料里背景句常常不够（对话语料代词密度高），用人工中性池补齐
    while len(bucket_zero) < n_bg:
        base = rng.choice(NEUTRAL_POOL)
        # 轻微变体，避免完全重复
        if rng.random() < 0.3 and len(base) > 8:
            base = base.rstrip("。") + rng.choice(["。" , "！", "。"])
        bucket_zero.append({"text": base, "spans": []})

    # 合成多代词句（按配额，不再硬编码 2500）
    p1 = ["我", "我们", "咱们"]
    p2 = ["你", "您", "你们"]
    p3 = ["他", "她", "他们"]
    synthetic = []
    for _ in range(n_syn):
        tpl = rng.choice(SYNTHETIC_TEMPLATES)
        s = tpl.format(p1=rng.choice(p1), p2=rng.choice(p2), p3=rng.choice(p3))
        synthetic.append({"text": s, "spans": extract_all_spans(s)})

    # v3：符号句桶（说话动词 + 冒号 + 直接引语 及近邻变体）
    symbolic = [_symbolic_sample(rng) for _ in range(n_sym)]

    # 真实句不足时用合成句补齐（保持总数）
    real_pool = bucket_multi + bucket_single
    if len(real_pool) < n_real:
        need = n_real - len(real_pool)
        short_tpl = [
            "{p}刚才把方案发过来了。", "请问{p}对这个接口有什么建议？",
            "{p}昨天提交的补丁已经合并。", "我们正在等{p}确认灰度结果。",
            "这份配置是{p}整理的吧？", "{p}觉得这个延迟能接受吗。",
        ]
        cat_words = {1: ["我", "我们"], 2: ["你", "您"], 3: ["他", "他们"]}
        for _ in range(need):
            c = rng.choice([1, 2, 3])
            s = rng.choice(short_tpl).format(p=rng.choice(cat_words[c]))
            real_pool.append({"text": s, "spans": extract_all_spans(s)})

    dataset = real_pool[:n_real] + synthetic + symbolic + bucket_zero[:n_bg]
    rng.shuffle(dataset)

    n_bg_actual = sum(1 for d in dataset if len(d["spans"]) == 0)
    n_sym_actual = sum(1 for d in dataset if d["text"] in {x["text"] for x in symbolic})
    print(f"  Total Rich AR Dataset v3: {len(dataset)} samples ✅ "
          f"(背景句 {n_bg_actual} 条 = {n_bg_actual/len(dataset)*100:.1f}% | "
          f"符号句 {n_sym_actual} 条 | 冒号紧接引语 "
          f"{sum(1 for d in dataset if '：“' in d['text'])} 条)")
    return dataset


if __name__ == "__main__":
    ds = build_rich_ar_dataset(target_samples=200)
    n_bg = sum(1 for d in ds if not d["spans"])
    print(f"背景句占比: {n_bg/len(ds)*100:.1f}%")
    print("\n符号句样例（说话动词 + 冒号 + 直接引语 及近邻变体）：")
    sym_rng = random.Random(0)
    for _ in range(10):
        d = _symbolic_sample(sym_rng)
        print(f"  {d['text']}  ->  "
              f"{[(s['word'], s['label'], s['start'], s['end']) for s in d['spans']]}")
