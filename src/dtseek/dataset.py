"""Dataset generator for DTSeek pronoun task with rich, diverse templates."""
import random

PRONOUN_MAP = {
    1: ["我", "我们", "咱们", "俺", "在下", "鄙人"],
    2: ["你", "你们", "您", "阁下"],
    3: ["他", "她", "它", "他们", "她们", "它们"],
}

# 确保无代词模板绝对干净（不包含任何“我你他她它”字）
NON_PRONOUN_TEMPLATES = [
    "今天天气非常晴朗，阳光明媚适合散步。",
    "春风又绿江南岸，明月何时照江山。",
    "数据库集群写入延迟保持在五毫秒以内。",
    "白日依山尽，黄河入海流。",
    "深度学习模型正在加速推理计算过程。",
    "红富士苹果富含维生素C，口感清甜多汁。",
    "高铁路网贯通南北，极大缩短了城际通勤时间。",
    "自动驾驶算法利用多传感器融合进行精准避障。",
    "图书馆周末正常开放，欢迎读者借阅书籍。",
    "晨曦初现，山林间弥漫着淡淡的薄雾。",
    "工业机器人按预设计划完成零件焊接组装。",
    "落霞与孤鹜齐飞，秋水共长天一色。",
    "机器学习通过数据统计规律提取特征。",
    "风吹草低见牛羊，大漠孤烟直。",
    "芯片制程工艺突破带来了算力大幅跃升。",
    "算法经过严谨推导保证了数学上的收敛性。",
]

CONTEXT_TEMPLATES = [
    "{pronoun}今天要去参加一个技术交流大会。",
    "听说明天下午开会，{pronoun}准备好汇报方案了吗？",
    "{pronoun}刚才提交了最新的代码合入请求。",
    "桌子上的那本书是{pronoun}借阅的，记得按期归还。",
    "在团队协作中，{pronoun}始终展现出积极严谨的态度。",
    "如果遇到技术问题，随时可以向{pronoun}咨询排查方案。",
    "{pronoun}对深度学习领域的模型结构演进非常感兴趣。",
    "项目测试全绿，这离不开{pronoun}这段时间的持续优化。",
    "{pronoun}和朋友约好周末一起看电影。",
    "关于这个方案的设计理念，{pronoun}有什么不同的见解？",
    "大家都很期待{pronoun}能分享这次架构重构的经验心得。",
    "{pronoun}已经在系统里配置好了监控告警策略。",
    "每次系统发布，{pronoun}都会仔细核对灰度流量比例。",
    "如果这件事情交给{pronoun}来负责，肯定能让人放心。",
    "昨晚加班排查故障，{pronoun}辛苦了。",
    "这次比赛中，{pronoun}发挥出色夺得了优异成绩。",
]


def generate_pronoun_dataset(num_samples: int = 2400) -> list[dict]:
    """Generates synthetic dataset balanced across classes 0, 1, 2, 3."""
    samples = []
    for _ in range(num_samples):
        target_class = random.choice([0, 1, 2, 3])
        if target_class == 0:
            text = random.choice(NON_PRONOUN_TEMPLATES)
        else:
            pronoun = random.choice(PRONOUN_MAP[target_class])
            tpl = random.choice(CONTEXT_TEMPLATES)
            text = tpl.format(pronoun=pronoun)

        samples.append({
            "text": text,
            "label": target_class,
        })
    return samples
