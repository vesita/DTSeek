"""构建对话发言人与归属主体（Speaker / Ownership Attribution）切片数据集。

任务体系：
  - 0: 无明确归属/客观事实 (背景类)
  - 1: 用户/客户/提问方 (用户、提问者、顾客、老哥、我方...)
  - 2: 助手/系统/客服 (模型、助手、系统、小助手、客服、机器人...)
  - 3: 第三方/外部人员/团队同事 (领导、老板、同事、架构师、专家、后端、前端...)

从真实对话日志中挖掘角色标识与归属声明。
"""
import glob
import random

SPEAKER_MAP = [
    # 类别 1: 用户/客户
    (1, ["用户", "提问者", "顾客", "买家", "客户", "提问方", "观众", "网友"]),
    # 类别 2: 助手/系统
    (2, ["模型", "助手", "小助手", "系统", "客服", "机器人", "程序", "服务端"]),
    # 类别 3: 第三方/团队成员
    (3, ["架构师", "工程师", "开发者", "负责人", "领导", "老板", "同事", "专家", "产品经理", "测试人员"]),
]

NEUTRAL_CORPUS = [
    "数据库集群写入延迟保持在五毫秒以内。",
    "自动驾驶算法利用多传感器融合进行精准避障。",
    "今天天气晴朗，气温约二十二度。",
    "白日依山尽，黄河入海流。",
    "高铁路网贯通南北，极大缩短了城际通勤时间。",
    "红富士苹果富含维生素C，口感清甜多汁。",
    # v2 补丁：归属人任务的训练数据几乎全是"用户：xxx / 模型：xxx"这种带角色前缀的行，
    # 模型因此没见过"裸句"（无任何角色标识）。演示时喂裸句就会瞎猜归属。
    # 这里补入裸口语句作为背景类，教会模型：没有角色标识 ⇒ 无归属。
    "我感觉效果好像有点问题。",
    "你好，你觉得你现在状态怎么样。",
    "这个方案的收益和成本怎么平衡。",
    "刚才那个接口的返回值好像变了。",
    "请问这个功能具体要怎么使用。",
    "文档里写的参数含义我看不太懂。",
    "这个报错信息对应什么原因。",
    "麻烦帮我核对一下这条记录。",
    "现在的进度到哪一步了。",
    "构建一次大概要多久。",
    "这段逻辑写得挺清楚的。",
    "我有点喜欢这个颜色。",
    "这个版本比上一版好用了不少。",
    "刚才的改动好像没生效。",
    "接下来需要做什么准备工作。",
]

DIALOGUE_TEMPLATES = [
    "{speaker}说：这个功能需要再评审一下。",
    "刚才{speaker}提到接口延迟有点偏高。",
    "请确认一下，这是{speaker}提交的需求吗？",
    "{speaker}刚才回复说问题已经解决了。",
    "关于这个架构设计，{speaker}给出了明确的改进方案。",
    "我们正在等待{speaker}确认灰度发布的观察结果。",
]


def extract_speaker_spans(text: str) -> list[dict]:
    spans = []
    occupied = [False] * len(text)

    flat_list = []
    for cat_id, words in SPEAKER_MAP:
        for w in words:
            flat_list.append((cat_id, w, len(w)))
    flat_list.sort(key=lambda x: -x[2])

    for cat_id, word, length in flat_list:
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
    return spans


def build_ownership_dataset(target_samples: int = 10000, max_seq_len: int = 64,
                            bg_ratio: float = 0.30, seed: int = 20240927) -> list[dict]:
    """构建归属人切片数据集。

    v1 缺陷：背景句只占 10%（`target_samples * 0.1`），且模板句里三类别严重不均，
    导致模型对无归属句同样 100% 误报。v2 把背景配额提到 30%，并显式对齐三类别。
    """
    rng = random.Random(seed)
    corpora_files = sorted(glob.glob("/home/vesita/coding/my/nanoSeek/data/chinese/*dialogue.txt"))

    n_bg = int(target_samples * bg_ratio)
    n_real = target_samples - n_bg

    print(f"  归属人数据集配额：真实句 {n_real} | 背景句 {n_bg}")

    dataset = []

    # 1. 真实对话行（含角色标识，如 "用户：..."）
    for f in corpora_files:
        if len(dataset) >= n_real:
            break
        with open(f, encoding="utf-8", errors="ignore") as fp:
            for line in fp:
                if len(dataset) >= n_real:
                    break
                s = line.strip()
                if 4 <= len(s) <= max_seq_len:
                    spans = extract_speaker_spans(s)
                    if spans:
                        dataset.append({"text": s, "spans": spans})

    # 2. 模板补齐（三类别轮转，保证类别均衡）
    template_idx = 0
    while len(dataset) < n_real:
        cat_id, words = SPEAKER_MAP[template_idx % len(SPEAKER_MAP)]
        template_idx += 1
        s = DIALOGUE_TEMPLATES[template_idx % len(DIALOGUE_TEMPLATES)].format(
            speaker=rng.choice(words))
        spans = extract_speaker_spans(s)
        if spans:
            dataset.append({"text": s, "spans": spans})

    # 3. 背景句（无任何归属标识）
    for _ in range(n_bg):
        s = rng.choice(NEUTRAL_CORPUS)
        dataset.append({"text": s, "spans": []})

    rng.shuffle(dataset)

    n_bg_actual = sum(1 for d in dataset if not d["spans"])
    print(f"归属人数据集构建完成：{len(dataset)} 样本 ✅ "
          f"(背景句 {n_bg_actual} = {n_bg_actual/len(dataset)*100:.1f}%)")
    return dataset


if __name__ == "__main__":
    ds = build_ownership_dataset(target_samples=20)
    for x in ds[:5]:
        print(f"'{x['text']}' -> spans: {[(s['word'], s['label'], s['start'], s['end']) for s in x['spans']]}")
