"""多代词真实语料 + 合成增强切片数据集（v2：显式配比、背景句不再被淹没）。

v1 的致命缺陷（已定位）：
    合成句硬编码 for _ in range(2500)，全部含代词、从不产背景句；
    且扫描循环在 bucket_multi 满时就 break，bucket_zero 常常根本没填满。
    实测 target=600 时背景句只占 4.7% —— 模型因此学会「永远开火」，
    对中性句 100% 误报。

v2 修复：四个桶各自独立填到配额（互不 early-break），并显式给背景句 32% 配额。
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


def build_rich_ar_dataset(target_samples: int = 15000, max_seq_len: int = 64,
                          bg_ratio: float = 0.32, synthetic_ratio: float = 0.20) -> list[dict]:
    """构建代词任务数据集。

    Args:
        target_samples: 目标总样本数
        bg_ratio: 背景句（无代词）配额比例，默认 32% —— v1 只有 4.7%，是误报根因
        synthetic_ratio: 合成多代词句配额比例
    """
    rng = random.Random(20240927)

    n_bg = int(target_samples * bg_ratio)
    n_syn = int(target_samples * synthetic_ratio)
    n_real = target_samples - n_bg - n_syn

    print(f"  代词数据集配额：真实句 {n_real} | 合成句 {n_syn} | 背景句 {n_bg}")

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

    dataset = real_pool[:n_real] + synthetic + bucket_zero[:n_bg]
    rng.shuffle(dataset)

    n_bg_actual = sum(1 for d in dataset if len(d["spans"]) == 0)
    print(f"  Total Rich AR Dataset v2: {len(dataset)} samples ✅ "
          f"(背景句 {n_bg_actual} 条 = {n_bg_actual/len(dataset)*100:.1f}%)")
    return dataset


if __name__ == "__main__":
    ds = build_rich_ar_dataset(target_samples=200)
    n_bg = sum(1 for d in ds if not d["spans"])
    print(f"背景句占比: {n_bg/len(ds)*100:.1f}%")
