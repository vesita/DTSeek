"""人物追踪（Person Tracking）数据集：按出场顺序给每个人物一个匿名 id。

任务形态
--------
一段文本里出现若干人物。按**首次出场顺序**给每人分配一个匿名 id：

    label 0 = 无人物（背景）
    label 1 = 第 1 个出场的人物
    label 2 = 第 2 个出场的人物
    ...
    label 7 = 第 7 个出场的人物

模型要对文本里**每一处人物提及**发射一个切片并给出该人物的 id —— 专名（张伟）、
称谓（王阿姨、张老师）、以及代词（他/她/我/你/您/我们/你们/他们/她们）。
同一人物的所有提及复用同一个 id。

为什么不做预设角色分类：类别数是「剧情里有几个人」决定的，不是先验的角色表。
模型必须自己把提及聚成人物簇，而不是把「他」映射到某个固定类别。

五条硬性设计取舍
----------------

1. **id 按首次出现严格递增。** 首次出现位置最小的 id=1，第二名 id=2，依此类推。
   校验时逐样本核对「按首次出现排序的 label 序列 == 1,2,3,...」，不允许跳号。

2. **代词指代必须唯一可判定，否则丢弃整条样本（fail-closed）。**
   生成时用真值约束、构建时用**独立于生成计划的解析器**复核（见下）。

3. **复数代词的 label 约定（本文件唯一一处人为约定，必须写清）。**
   输出形状是「一个 span → 一个 label」，而 他们/她们/我们/你们 指代的是一**组**
   人（≥2 个 id），单个 label 表达不了。本文件的约定是：

       **复数代词取「所指集合中 id 最小者」的 label。**

   为让这条约定可学，生成时保证所指集合能被前文**唯一**确定：
     - 他们/她们：同句前文明确枚举该组（`A、B和C一起…，他们…`），且带集体标记
       （一起/一块儿/都/俩）；性别一致（他们 = 非全女，她们 = 全女）；
     - 我们：只出现在 `A对B说：“我们…”` 这类说话框架里，指 {说话人} ∪ {受话人}；
     - 你们：只出现在 `A对B和C说：“你们…”` 里，指明确枚举的受话人集合，且 ≥2 人。
   任何一条推不出唯一集合的复数代词样本都会被复核器丢弃。

4. **>7 个人物 → 丢弃整条样本**（不截断文本）。截断会让「按出场顺序编号」这条
   规则的边界变得不可解释（截断后某人的首次出现可能落在截断点之后）。
   生成器本身只造 2..7 人，所以真实语料挖掘才可能触发这条。

5. **真实语料只用于背景，人物切片 100% 合成。**
   实测（1.5M 行对话语料）：名字库里出现过的名字共命中 ~304 次，
   而「≥2 个库里名字 + 无任何代词」的可验证片段是 **0 条**。真实片段里只要出现
   一个库外的人名/称谓（妈妈、记者、老王…）就会漏标 —— 而漏标就是错监督，
   比多几百条合成样本贵得多。所以真实语料只贡献背景（零人物词的中性句），
   人物切片全部由受控模板合成，真值由构造保证、再由独立复核器验证。

直接引语（本任务的核心价值）
----------------------------
`A对B说：“我明天来找你。”` 里引号内的「我」指说话人 A、「你」指受话人 B；
`A问B：“你住在哪里？”` 里受话人在动词**之后**。这些都由 `_quote_frames` 独立解析
（找说话动词 → 取动词前的说话人、`对/向/跟` 后的受话人、动词后的受话人 → 引语体），
解析不出唯一说话人/受话人的样本一律丢弃。

失败即丢弃（fail-closed），不静默产出：`_check_sample` 是逐样本闸门，
`_validate_sample` 是构建末尾的硬断言。
"""
from __future__ import annotations

import glob
import random
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

# ── 名字库 ────────────────────────────────────────────────────────────────
# 常见中文全名，无生僻字；性别可由名字库本身判定（用于性别一致的代词约束）。
MALE_NAMES = (
    "张伟", "王强", "李军", "刘洋", "陈杰", "杨磊", "赵鹏", "黄涛",
    "周斌", "吴刚", "徐峰", "孙浩", "马超", "朱亮", "胡兵", "郭勇",
    "林伟", "何军", "高翔", "罗刚", "郑凯", "梁辉", "谢东", "宋健",
)
FEMALE_NAMES = (
    "王芳", "李娜", "张敏", "刘婷", "陈静", "杨丽", "赵敏", "黄燕",
    "周雪", "吴倩", "徐丽", "孙悦", "马丽", "朱琳", "胡蝶", "郭静",
    "林娟", "何梅", "高媛", "罗琳", "郑爽", "梁静", "谢娜", "宋佳",
)

#: 名字库里用到的全部姓氏（供探针构造「库里名字 vs 非名字」对照）
PERSON_SURNAMES = tuple(sorted({n[0] for n in MALE_NAMES + FEMALE_NAMES}))

#: 无性别称谓/角色 —— 只能被专名/称谓提及，**不能被 他/她 指代**
#: （性别未知 ⇒ 代词指代不可能唯一）
NEUTRAL_ROLES = ("老师", "医生")

_HONORIFIC_GENDER = {
    "叔叔": "m", "爷爷": "m", "先生": "m",
    "阿姨": "f", "奶奶": "f", "女士": "f",
    "老师": None, "医生": None,
}
HONORIFIC_SURFACES = tuple(f"{s}{suf}" for s in PERSON_SURNAMES for suf in _HONORIFIC_GENDER)

# ── 代词 ──────────────────────────────────────────────────────────────────
_SINGULAR_THIRD = {"他": "m", "她": "f"}
_FIRST_PERSON = ("我",)
_SECOND_PERSON = ("你", "您")
#: 复数第三人称 → 允许的组性别构成（"not_all_f" = 多个男性或混合；"all_f" = 全女）
_PLURAL_PRONOUNS = {"他们", "她们", "我们", "咱们", "你们", "它们"}  # 只用于 fail-closed 拒绝，不参与生成
_FIRST_SECOND_PRONOUNS = set(_FIRST_PERSON) | set(_SECOND_PERSON)

#: 说话动词。**故意不含「道」与单字「答」** —— 「知道」「报道」「难道」「答应」
#: 会把它们撞成假引语框架，而假框架会让独立复核器误判说话人。
_SPEECH_VERBS = ("回答", "告诉", "说", "问", "喊")
_VERB_RE = re.compile("|".join(sorted(_SPEECH_VERBS, key=len, reverse=True)))

#: 背景句里**不允许**出现的人物名词。词表只覆盖专名/称谓/代词，但
#: 「妈妈、记者、朋友」同样是人物的提及 —— 背景句含它们就是漏标，必须挡掉。
_PERSON_NOUNS = (
    "妈妈", "爸爸", "母亲", "父亲", "儿子", "女儿", "孩子", "朋友", "同事", "同学",
    "同桌", "邻居", "亲戚", "家人", "老人", "男孩", "女孩", "男人", "女人", "记者",
    "专家", "医生", "护士", "老师", "学生", "司机", "警察", "工人", "农民", "顾客",
    "客户", "老板", "领导", "经理", "工程师", "服务员", "运动员", "演员", "歌手",
    "作家", "画家", "人们", "大家", "有人", "别人", "本人", "对方", "双方", "各位",
    "诸位", "师傅", "阿姨", "叔叔", "爷爷", "奶奶", "哥哥", "姐姐", "弟弟", "妹妹",
    "女士", "先生", "夫人", "太太", "小姐", "成员",
)
_PERSON_NOUN_RE = re.compile("|".join(sorted(_PERSON_NOUNS, key=len, reverse=True)))

# ── 人物词表（用于「有没有漏标」的独立扫描） ──────────────────────────────
_FULL_NAMES = frozenset(MALE_NAMES + FEMALE_NAMES)
_ALIASES = tuple(
    {f"{pre}{s}" for s in PERSON_SURNAMES for pre in ("小", "老")}
)


def _build_vocab() -> dict[str, str]:
    vocab: dict[str, str] = {}
    for n in MALE_NAMES + FEMALE_NAMES:
        vocab[n] = "name"
    for s in HONORIFIC_SURFACES:
        vocab[s] = "honorific"
    for r in NEUTRAL_ROLES:
        vocab[r] = "role"
    for a in _ALIASES:
        vocab[a] = "alias"
    for p in _SINGULAR_THIRD:
        vocab[p] = "pronoun"
    for p in _FIRST_PERSON + _SECOND_PERSON + ():
        vocab[p] = "pronoun"
    return vocab


_PERSON_VOCAB = _build_vocab()
_VOCAB_BY_FIRST: dict[str, tuple[str, ...]] = {}
for _s in _PERSON_VOCAB:
    _VOCAB_BY_FIRST.setdefault(_s[0], ())
    _VOCAB_BY_FIRST[_s[0]] = tuple(sorted(_VOCAB_BY_FIRST[_s[0]] + (_s,), key=len, reverse=True))


# ── 语料 ──────────────────────────────────────────────────────────────────
CORPUS_GLOB = "/home/vesita/coding/my/nanoSeek/data/chinese/*dialogue.txt"
_ROLE_PREFIX = re.compile(r"^(用户|模型|系统|提问|回答|User|Assistant)[:：]\s*")
#: 切句时**保留**终止标点 —— 否则背景句全都没有句号，模型会用「有没有句号」当捷径
_SENT_SPLIT = re.compile(r"[^。！？\n；;]+[。！？]?")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_BG_JUNK = "：:*<>[]{}`|#"

#: 百家姓常用字。真实语料里会出现词表覆盖不到的**人名**（昵称「雪雪」、
#: 「张涛」…）—— 背景句一旦含它们就是漏标。词表查不到的名字用两条启发式挡：
#:   1) 姓氏字 + 紧跟一个汉字（疑似姓名）；
#:   2) 汉字叠字（疑似昵称，如「雪雪」「萌萌」）。
#: 这两条会误杀一部分干净背景（「高兴」「马上」「刚刚」…），但真实语料足够大，
#: 用覆盖率换监督纯度是划算的。
_COMMON_SURNAMES = frozenset(
    "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜戚谢邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳鲍史唐费廉岑薛雷贺倪汤滕殷罗毕郝邬安常乐于时傅皮齐康伍余元卜顾孟平黄和穆萧尹姚邵汪祁毛禹狄米贝明臧计伏成戴谈宋茅庞熊纪舒屈项祝董梁杜阮蓝闵席季麻强贾路娄危江童颜郭梅盛林刁钟徐邱骆高夏蔡田樊胡凌霍虞万支柯管卢莫房裘缪干解应宗丁宣邓郁单杭洪包诸左石崔吉钮龚程嵇邢滑裴陆荣翁荀羊甄曲封芮羿储靳汲邴糜松井段富巫乌焦巴弓牧隗山谷车侯宓蓬全郗班仰秋仲伊宫宁仇栾暴甘钭厉戎祖武符刘景詹束龙叶幸司韶郜黎蓟薄印宿白怀蒲邰从鄂索咸籍赖卓蔺屠蒙池乔阴胥能苍双闻莘党翟谭贡劳逄姬申扶堵冉宰郦雍却璩桑桂濮牛寿通边扈燕冀郏浦尚农温别庄晏柴瞿阎充慕连茹习宦艾鱼容向古易慎戈廖庾终暨居衡步都耿满弘匡国文寇广禄阙东欧殳沃利蔚越夔隆师巩厍聂晁勾敖融冷訾辛阚那简饶空曾毋沙乜养鞠须丰巢关蒯相查后荆红游竺权逯盖益桓公"
)
_REDUP_RE = re.compile(r"([\u4e00-\u9fff])\1")


def _looks_like_person_name(s: str) -> str | None:
    """返回命中的启发式名（'姓氏' / '叠字'），没命中返回 None。"""
    for i in range(len(s) - 1):
        if s[i] in _COMMON_SURNAMES and _CJK_RE.match(s[i + 1]):
            return "姓氏"
    if _REDUP_RE.search(s):
        return "叠字"
    return None


def _bg_quality_ok(s: str) -> bool:
    """真实语料里挖出来的句子必须先过质量闸门，否则会混进 markdown/代码输出碎片。"""
    if len(_CJK_RE.findall(s)) < 4:
        return False
    if any(c in s for c in _BG_JUNK):
        return False
    first = s[0]
    if not ("\u4e00" <= first <= "\u9fff") and not first.isdigit():
        return False
    return _looks_like_person_name(s) is None


# ══════════════════════════════════════════════════════════════════════════
# 独立扫描器 / 校验器
# ══════════════════════════════════════════════════════════════════════════
def _match_vocab_at(text: str, i: int) -> dict | None:
    """在 text[i] 处做最长匹配，命中人物词表则返回提及；否则 None。"""
    for surf in _VOCAB_BY_FIRST.get(text[i], ()):
        if text.startswith(surf, i):
            return {"word": surf, "start": i, "end": i + len(surf), "kind": _PERSON_VOCAB[surf]}
    return None


def _scan_person_mentions(text: str) -> list[dict]:
    """最长匹配、不重叠地扫出文本里所有人物提及。

    这是「有没有漏标」的独立口径：它不认识生成计划，只认词表。
    """
    out: list[dict] = []
    i = 0
    while i < len(text):
        m = _match_vocab_at(text, i)
        if m is None:
            i += 1
        else:
            out.append(m)
            i = m["end"]
    return out


def _mentions_in(scan: list[dict], lo: int, hi: int) -> list[dict]:
    return [m for m in scan if m["start"] >= lo and m["end"] <= hi]


def _end_of_sentence(text: str, start: int) -> int:
    k = start
    while k < len(text) and text[k] not in "。！？":
        k += 1
    return k


def _sentence_start(text: str, pos: int) -> int:
    start = 0
    for c in "。！？":
        k = text.rfind(c, 0, pos)
        if k + 1 > start:
            start = k + 1
    return start


def _quote_frames(text: str, scan: list[dict]) -> list[dict]:
    """把文本解析成「说话框架」列表，用于独立判定 我/你/我们/你们 的指代。

    每个框架给出：引语体的 [body_start, body_end)、说话人提及、受话人提及。
    解析不出的（没有受话人、头部混入了别的引语等）就留给复核器判「指代不唯一」。
    """
    frames: list[dict] = []
    for m in _VERB_RE.finditer(text):
        v_start, v_end = m.start(), m.end()
        sent_start = _sentence_start(text, v_start)
        head = text[sent_start:v_start]
        if "“" in head or '"' in head:
            # 头部混进了别的引语 ⇒ 这不是一个干净的说话框架（可能是引语内部的动词）
            continue
        # 动词与引语体之间：允许标点 + 受话人；但遇到 我/你 类代词就停（那是引语内容）
        j = v_end
        saw_colon = False
        between: list[dict] = []
        while j < len(text):
            c = text[j]
            if c in " \u3000：:，,":
                if c in "：:":
                    saw_colon = True
                j += 1
                continue
            mm = _match_vocab_at(text, j)
            if mm is None:
                break
            if mm["word"] in _FIRST_SECOND_PRONOUNS:
                break
            between.append(mm)
            j = mm["end"]
        if j < len(text) and text[j] in "“\"":
            body_start = j + 1
            close = text.find("”", body_start)
            body_end = close if close >= 0 else len(text)
        else:
            body_start = j
            body_end = _end_of_sentence(text, j)
            if not saw_colon and not between and text[v_end:v_end + 1] not in ("，", ","):
                # 动词后既没有标点也没有受话人 —— 不是引语（如「他问路」里的「问」）
                if body_start == v_end:
                    continue
        head_mentions = _mentions_in(scan, sent_start, v_start)
        marker = -1
        for ch in "对向跟":
            k = head.rfind(ch)
            if k > marker:
                marker = k
        # marker 是 head 内相对下标，head_mentions 是绝对坐标 —— 必须换算
        marker_abs = sent_start + marker if marker >= 0 else -1
        if marker_abs >= 0:
            speakers = [mm for mm in head_mentions if mm["end"] <= marker_abs]
            addrs = [mm for mm in head_mentions if mm["start"] > marker_abs]
        else:
            speakers = head_mentions
            addrs = []
        addrs = addrs + between
        frames.append({
            "body_start": body_start, "body_end": body_end,
            "speakers": speakers, "addrs": addrs, "verb": m.group(0),
        })
    return frames


# ── 构造期的真值包 ────────────────────────────────────────────────────────
@dataclass
class _Truth:
    pid_gender: dict[int, str | None]
    pid_surfaces: dict[int, set[str]]
    n_persons: int


def _surface_to_label(spans: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for s in spans:
        if _PERSON_VOCAB.get(s["word"], "pronoun") != "pronoun":
            out[s["word"]] = s["label"]
    return out


def _resolve_pronoun(text: str, mention: dict, scan: list[dict],
                     surf2label: dict[str, int], spans: list[dict],
                     truth: _Truth) -> set[int]:
    """独立推一遍这个代词**可能**指代谁，返回候选 label 集合。

    调用方要求 len(候选) == 1，否则丢弃整条样本。
    """
    word = mention["word"]
    pos = mention["start"]
    first_start: dict[int, int] = {}
    for s in spans:
        first_start.setdefault(s["label"], s["start"])

    if word in _SINGULAR_THIRD:
        g = _SINGULAR_THIRD[word]
        return {pid for pid, pg in truth.pid_gender.items()
                if pg == g and first_start.get(pid, 10 ** 9) < pos}

    if word in ("他们", "她们"):
        sent_start = _sentence_start(text, pos)
        members = [m for m in _mentions_in(scan, sent_start, pos)
                   if _PERSON_VOCAB.get(m["word"]) != "pronoun"]
        labels = {surf2label[m["word"]] for m in members if m["word"] in surf2label}
        if len(labels) < 2:
            return set()
        cue = any(c in text[sent_start:pos] for c in ("一起", "一块儿", "一同", "都", "俩"))
        if not cue:
            return set()
        genders = {truth.pid_gender[pid] for pid in labels}
        if word == "她们" and genders != {"f"}:
            return set()
        if word == "他们" and genders == {"f"}:
            return set()
        return {min(labels)}

    # 我 / 你 / 您 / 我们 / 你们 → 说话框架
    frames = _quote_frames(text, scan)
    covering = [f for f in frames if f["body_start"] <= pos < f["body_end"]]
    if not covering:
        return set()
    if word in ("我", "我们", "咱们"):
        spk = {surf2label[m["word"]] for f in covering for m in f["speakers"]
               if m["word"] in surf2label}
        if len(spk) != 1:
            return set()
        if word == "我":
            return spk
        addrs = {surf2label[m["word"]] for f in covering for m in f["addrs"]
                 if m["word"] in surf2label}
        if not addrs:
            return set()
        return {min(spk | addrs)}
    # 你 / 您 / 你们
    addrs = {surf2label[m["word"]] for f in covering for m in f["addrs"]
             if m["word"] in surf2label}
    if word in ("你", "您"):
        return addrs
    if len(addrs) < 2:
        return set()
    return {min(addrs)}


def _check_sample(text: str, spans: list[dict], truth: _Truth | None = None,
                  max_len: int = 120) -> str | None:
    """逐样本闸门：返回 None 表示这条样本可以进数据集，否则返回丢弃原因。"""
    if not (4 <= len(text) <= max_len):
        return "长度越界"
    if not spans:
        return "正例无切片"
    prev_end = -1
    for s in spans:
        if not isinstance(s["label"], int) or not 1 <= s["label"] <= 7:
            return "label 越界"
        if not (0 <= s["start"] < s["end"] <= len(text)):
            return "区间越界"
        if text[s["start"]:s["end"]] != s["word"]:
            return "span 字面与原文不符"
        if s["start"] < prev_end:
            return "span 重叠"
        prev_end = s["end"]

    scan = _scan_person_mentions(text)
    if [(m["start"], m["end"], m["word"]) for m in scan] != \
       [(s["start"], s["end"], s["word"]) for s in spans]:
        return "漏标或错标的人物提及"

    # id 必须按首次出现顺序 1,2,3,...
    first: dict[int, int] = {}
    for s in spans:
        first.setdefault(s["label"], s["start"])
    by_pos = sorted(first.items(), key=lambda kv: kv[1])
    if [pid for pid, _ in by_pos] != list(range(1, len(first) + 1)):
        return "id 未按首次出现顺序"

    if truth is None:
        return None

    if truth.n_persons != len(first):
        return "有未出场的人物"
    if any(pid == 0 for pid in truth.pid_gender):
        return "存在从未被提及的人物"

    surf2label = _surface_to_label(spans)
    for s in spans:
        kind = _PERSON_VOCAB.get(s["word"], "pronoun")
        if kind != "pronoun" and s["word"] not in truth.pid_surfaces.get(s["label"], set()):
            return "提及与人物真值不符"

    for m in scan:
        if _PERSON_VOCAB.get(m["word"]) != "pronoun":
            continue
        cands = _resolve_pronoun(text, m, scan, surf2label, spans, truth)
        if len(cands) != 1:
            return f"代词指代不唯一:{m['word']}"
        # 该代词所在切片的 label 必须等于唯一候选
        lbl = next(s["label"] for s in spans if s["start"] == m["start"])
        if lbl != next(iter(cands)):
            return f"代词指代与标注不符:{m['word']}"
    return None


_PLURAL_RE = re.compile("|".join(sorted(_PLURAL_PRONOUNS, key=len, reverse=True)))


def _reject_plural(text: str) -> None:
    """复数提及其正确答案是集合，本任务的输出形状表达不了 —— 既然不标，就不许出现。"""
    hit = _PLURAL_RE.search(text)
    if hit:
        raise ValueError(f"文本含复数代词 {hit.group()!r}，本任务不标注复数提及：{text!r}")


def _plural_hit(text: str) -> str | None:
    hit = _PLURAL_RE.search(text)
    return hit.group() if hit else None


def _validate_sample(text: str, spans: list[dict], truth: _Truth | None = None,
                     max_len: int = 120) -> None:
    """硬断言版校验：出问题直接抛错（构建末尾与自检用）。"""
    hit = _plural_hit(text)
    if hit:
        raise ValueError(f"person 样本含复数代词 {hit!r}（本任务不标注复数提及）: {text!r}")
    reason = _check_sample(text, spans, truth, max_len)
    if reason is not None:
        raise ValueError(f"person 样本校验失败[{reason}]: {text!r} -> {spans}")


def _check_background(text: str, max_len: int = 120) -> str | None:
    """背景样本必须真的没有任何人物提及。"""
    if not (4 <= len(text) <= max_len):
        return "长度越界"
    if _plural_hit(text):
        return f"背景句含复数代词:{_plural_hit(text)}"
    if _scan_person_mentions(text):
        return "背景句含人物词"
    hit = _PERSON_NOUN_RE.search(text)
    if hit:
        return f"背景句含人物名词:{hit.group(0)}"
    return None


# ══════════════════════════════════════════════════════════════════════════
# 生成器
# ══════════════════════════════════════════════════════════════════════════
@dataclass
class _Person:
    surface: str
    gender: str | None
    alias: str | None = None
    pid: int = 0
    canonical_used: bool = False


@dataclass
class _Piece:
    text: str
    label_pid: int | None = None      # 该提及对应的 label（复数代词 = 组内最小 id）


class _PidAllocator:
    """按**提及顺序**发 id：第一次被提及的人拿 1，第二个拿 2，依此类推。"""

    def __init__(self) -> None:
        self.next_pid = 1

    def of(self, p: _Person) -> int:
        if p.pid == 0:
            p.pid = self.next_pid
            self.next_pid += 1
        return p.pid


_NAME_POOL: list[tuple[str, str, str | None]] = []
for _n in MALE_NAMES:
    _NAME_POOL.append((_n[0], _n, "m"))
for _n in FEMALE_NAMES:
    _NAME_POOL.append((_n[0], _n, "f"))
for _s in PERSON_SURNAMES:
    for _suf, _g in _HONORIFIC_GENDER.items():
        _NAME_POOL.append((_s, _s + _suf, _g))
for _r in NEUTRAL_ROLES:
    _NAME_POOL.append((_r[0], _r, None))


def _gender_plan(rng: random.Random, n: int) -> list[str | None]:
    """选一种性别构成。刻意让「同性别唯一」经常成立（他/她 才有覆盖），
    同时保留「2 男 2 女」这类没有任何单数代词的场景 —— 逼模型靠专名而不是靠代词捷径。"""
    # m_only / m_f_f 被刻意加权：它们同时让「他」性别唯一、又让「她们」有用武之地
    kinds = ["both", "m_only", "m_only", "m_f_f", "f_only", "m_neutral", "f_neutral", "mixed"]
    kind = rng.choice(kinds)
    if kind == "both":
        return ["m", "f"] + [None] * (n - 2)
    if kind == "m_only":
        return ["m"] + ["f"] * (n - 1)
    if kind == "m_f_f" and n >= 3:
        return ["m", "f", "f"] + [None] * (n - 3)
    if kind == "f_only":
        return ["f"] + ["m"] * (n - 1)
    if kind == "m_neutral":
        return ["m"] + [None] * (n - 1)
    if kind == "f_neutral":
        return ["f"] + [None] * (n - 1)
    if n >= 4:
        return ["m", "m", "f", "f"] + [None] * (n - 4)
    return ["m", "f"] + [None] * (n - 2)


def _pick_persons(rng: random.Random, genders: list[str | None]) -> list[_Person] | None:
    pool = list(_NAME_POOL)
    rng.shuffle(pool)
    used_first: set[str] = set()
    persons: list[_Person] = []
    for g in genders:
        for first, surf, pg in pool:
            if pg == g and first not in used_first:
                used_first.add(first)
                persons.append(_Person(surface=surf, gender=pg))
                break
        else:
            return None
    for p in persons:
        if p.surface in _FULL_NAMES:
            pre = "小" if (p.gender == "f" or rng.random() < 0.7) else "老"
            p.alias = pre + p.surface[0]
    return persons


def _name_piece(p: _Person, rng: random.Random, alloc: _PidAllocator) -> _Piece:
    """专名/称谓提及。别名只在专名已经出现过之后才用（否则无从判定同指）。"""
    pid = alloc.of(p)
    if p.alias and p.canonical_used and rng.random() < 0.4:
        return _Piece(p.alias, pid)
    p.canonical_used = True
    return _Piece(p.surface, pid)


def _gender_piece(p: _Person, alloc: _PidAllocator) -> _Piece:
    """他/她。只能是**已经出场**且性别唯一的人 —— 由调用方保证性别唯一。"""
    assert p.pid != 0, "代词不能先于该人物的专名出现"
    assert p.gender in ("m", "f"), "无性别称谓不能被 他/她 指代"
    return _Piece("他" if p.gender == "m" else "她", p.pid)


def _first_piece(p: _Person, alloc: _PidAllocator, word: str = "我") -> _Piece:
    assert p.pid != 0
    return _Piece(word, p.pid)


def _fill(tpl: str, mapping: dict[str, _Piece]) -> list[_Piece]:
    out: list[_Piece] = []
    pos = 0
    for m in re.finditer(r"\{([A-Za-z0-9_]+)\}", tpl):
        if m.start() > pos:
            out.append(_Piece(tpl[pos:m.start()]))
        out.append(mapping[m.group(1)])
        pos = m.end()
    if pos < len(tpl):
        out.append(_Piece(tpl[pos:]))
    return out


def _pieces_text(pieces: list[_Piece]) -> str:
    return "".join(p.text for p in pieces)


# ── 模板池（集中声明，便于体检「模板字面里混入人物名词」） ────────────────
TPL_INTRO1 = (
    "{A}今天来得特别早。",
    "{A}昨天刚从外地回来。",
    "{A}最近在准备一场考试。",
)
TPL_INTRO2 = (
    "{A}和{B}一起走进了教室。",
    "{A}把一本小说借给了{B}。",
    "{A}和{B}住在同一个小区。",
    "{A}昨天在图书馆遇到了{B}。",
    "{A}和{B}约好周末去看展览。",
)
TPL_INTRO3 = (
    "{A}、{B}和{C}一起去了公园。",
    "{A}、{B}和{C}约好周末去爬山。",
    "{A}、{B}和{C}在同一个兴趣小组。",
)
TPL_INTRO4 = (
    "{A}、{B}、{C}和{D}都在同一个小组。",
)
TPL_ATTR = (
    "{A}喜欢数学。",
    "{A}昨天买了一本新书。",
    "{A}对摄影很感兴趣。",
    "{A}的成绩一直很好。",
    "{A}在准备下周的演讲。",
    "{A}把课桌收拾得很整齐。",
)
TPL_ATTR2 = (
    "{A}和{B}一起报名了比赛。",
    "{A}把笔记借给了{B}。",
    "{A}约{B}周末去打球。",
)
TPL_ANAPHORA = (
    "{He}喜欢数学。",
    "{He}把作业交了。",
    "{He}住在学校附近。",
    "{He}昨天生病了。",
    "{He}一直在准备比赛。",
)
TPL_PAIR_MIX = (
    "{A}和{B}坐在一排，{HeA}喜欢数学，{SheB}喜欢语文。",
)
TPL_REVERSE = (
    "{A}把书借给了{B}，{ProB}同意明天还给{ProA}。",
    "{B}问{ProA}住在哪里，{ProA}说{ProB}可以来看看。",
)
TPL_QUOTE_SIMPLE = (
    "{A}说：“{I}明天不来了。”",
    "{A}说：“{I}已经吃过饭了。”",
)
TPL_QUOTE_ADDR = (
    "{A}对{B}说：“{I}明天来找{You}。”",
    "{A}问{B}：“{You}住在哪里？”",
    "{A}告诉{B}：“{I}明天把书还给{You}。”",
    "{A}回答{B}：“{I}知道了。”",
    "{A}对{B}说：“{I}把这件事记下来了。”",
)
TPL_QUOTE_THIRD = (
    "{A}对{B}说：“{HeC}昨天来找过{You}。”",
    "{A}告诉{B}：“{HeC}明天会来。”",
)
TPL_QUOTE_COLON = (
    "{A}对{B}说：{I}明天来找{You}。",
    "{A}告诉{B}：{I}明天就走。",
)
TPL_QUOTE_NOCOLON = (
    "{A}对{B}说“{I}明天来找{You}”。",
)
TPL_QUOTE_COMMA = (
    "{A}告诉{B}，{I}明天去找{You}。",
    "{A}对{B}说，{I}马上就回来。",
)
TPL_PLURAL3 = (
    "{A}、{B}和{C}一起去了公园，{They}带了很多东西。",
    "{A}、{B}和{C}一块儿去看展览，{They}都很开心。",
)
TPL_PLURAL2 = (
    "{A}和{B}一块儿去看展览，{They}都很开心。",
)
TPL_QUOTE_WE = (
    "{A}对{B}说：“{We}明天一起去吧。”",
)
TPL_QUOTE_YOUPL = (
    "{A}对{B}和{C}说：“{YouPl}先走，{I}随后就到。”",
)

_ALL_TPL = (
    TPL_INTRO1 + TPL_INTRO2 + TPL_INTRO3 + TPL_INTRO4 + TPL_ATTR + TPL_ATTR2
    + TPL_ANAPHORA + TPL_PAIR_MIX + TPL_REVERSE + TPL_QUOTE_SIMPLE + TPL_QUOTE_ADDR
    + TPL_QUOTE_THIRD + TPL_QUOTE_COLON + TPL_QUOTE_NOCOLON + TPL_QUOTE_COMMA
    + TPL_PLURAL3 + TPL_PLURAL2 + TPL_QUOTE_WE + TPL_QUOTE_YOUPL
)


def audit_templates() -> dict:
    """静态体检：模板字面里是否混入了人物名词（会变成漏标）。"""
    bad = []
    for tpl in _ALL_TPL:
        literal = re.sub(r"\{[A-Za-z0-9_]+\}", "", tpl)
        hit = _PERSON_NOUN_RE.search(literal)
        if hit:
            bad.append((tpl, hit.group(0)))
    return {"person_noun_leaks": bad}


# ── 子句生成 ──────────────────────────────────────────────────────────────
def _cl_intro1(p: _Person, rng: random.Random, alloc: _PidAllocator) -> list[_Piece]:
    return _fill(rng.choice(TPL_INTRO1), {"A": _name_piece(p, rng, alloc)})


def _cl_intro2(a: _Person, b: _Person, rng: random.Random, alloc: _PidAllocator) -> list[_Piece]:
    return _fill(rng.choice(TPL_INTRO2),
                 {"A": _name_piece(a, rng, alloc), "B": _name_piece(b, rng, alloc)})


def _cl_intro3(a: _Person, b: _Person, c: _Person, rng: random.Random,
               alloc: _PidAllocator) -> list[_Piece]:
    return _fill(rng.choice(TPL_INTRO3),
                 {"A": _name_piece(a, rng, alloc), "B": _name_piece(b, rng, alloc),
                  "C": _name_piece(c, rng, alloc)})


def _cl_intro4(group: list[_Person], rng: random.Random, alloc: _PidAllocator) -> list[_Piece]:
    keys = ["A", "B", "C", "D"]
    return _fill(rng.choice(TPL_INTRO4),
                 {k: _name_piece(p, rng, alloc) for k, p in zip(keys, group, strict=True)})


def _cl_attr(p: _Person, rng: random.Random, alloc: _PidAllocator) -> list[_Piece]:
    return _fill(rng.choice(TPL_ATTR), {"A": _name_piece(p, rng, alloc)})


def _cl_attr2(a: _Person, b: _Person, rng: random.Random, alloc: _PidAllocator) -> list[_Piece]:
    return _fill(rng.choice(TPL_ATTR2),
                 {"A": _name_piece(a, rng, alloc), "B": _name_piece(b, rng, alloc)})


def _cl_anaphora(p: _Person, rng: random.Random, alloc: _PidAllocator) -> list[_Piece]:
    return _fill(rng.choice(TPL_ANAPHORA), {"He": _gender_piece(p, alloc)})


def _cl_pair_mix(a: _Person, b: _Person, rng: random.Random, alloc: _PidAllocator) -> list[_Piece]:
    return _fill(rng.choice(TPL_PAIR_MIX),
                 {"A": _name_piece(a, rng, alloc), "B": _name_piece(b, rng, alloc),
                  "HeA": _gender_piece(a, alloc), "SheB": _gender_piece(b, alloc)})


def _cl_reverse(a: _Person, b: _Person, rng: random.Random, alloc: _PidAllocator) -> list[_Piece]:
    return _fill(rng.choice(TPL_REVERSE),
                 {"A": _name_piece(a, rng, alloc), "B": _name_piece(b, rng, alloc),
                  "ProA": _gender_piece(a, alloc), "ProB": _gender_piece(b, alloc)})


def _cl_quote_simple(a: _Person, rng: random.Random, alloc: _PidAllocator) -> list[_Piece]:
    return _fill(rng.choice(TPL_QUOTE_SIMPLE),
                 {"A": _name_piece(a, rng, alloc), "I": _first_piece(a, alloc)})


def _cl_quote_addr(a: _Person, b: _Person, rng: random.Random,
                   alloc: _PidAllocator) -> list[_Piece]:
    tpl = rng.choice(TPL_QUOTE_ADDR + TPL_QUOTE_COLON + TPL_QUOTE_NOCOLON + TPL_QUOTE_COMMA)
    return _fill(tpl, {"A": _name_piece(a, rng, alloc), "B": _name_piece(b, rng, alloc),
                       "I": _first_piece(a, alloc),
                       "You": _first_piece(b, alloc, rng.choice(("你", "您")))})


def _cl_quote_third(a: _Person, b: _Person, c: _Person, rng: random.Random,
                    alloc: _PidAllocator) -> list[_Piece]:
    return _fill(rng.choice(TPL_QUOTE_THIRD),
                 {"A": _name_piece(a, rng, alloc), "B": _name_piece(b, rng, alloc),
                  "HeC": _gender_piece(c, alloc),
                  "You": _first_piece(b, alloc, rng.choice(("你", "您")))})


def _cl_plural(group: list[_Person], rng: random.Random, alloc: _PidAllocator) -> list[_Piece]:
    all_female = all(p.gender == "f" for p in group)
    word = "她们" if all_female else "他们"
    rep = min(alloc.of(p) for p in group)
    if len(group) >= 3:
        tpl = rng.choice(TPL_PLURAL3)
        return _fill(tpl, {"A": _name_piece(group[0], rng, alloc),
                           "B": _name_piece(group[1], rng, alloc),
                           "C": _name_piece(group[2], rng, alloc),
                           "They": _Piece(word, rep)})
    tpl = rng.choice(TPL_PLURAL2)
    return _fill(tpl, {"A": _name_piece(group[0], rng, alloc),
                       "B": _name_piece(group[1], rng, alloc),
                       "They": _Piece(word, rep)})


def _cl_quote_we(a: _Person, b: _Person, rng: random.Random, alloc: _PidAllocator) -> list[_Piece]:
    rep = min(alloc.of(a), alloc.of(b))
    return _fill(rng.choice(TPL_QUOTE_WE),
                 {"A": _name_piece(a, rng, alloc), "B": _name_piece(b, rng, alloc),
                  "We": _Piece(rng.choice(("我们", "咱们")), rep)})


def _cl_quote_youpl(a: _Person, b: _Person, c: _Person, rng: random.Random,
                    alloc: _PidAllocator) -> list[_Piece]:
    rep = min(alloc.of(b), alloc.of(c))
    return _fill(rng.choice(TPL_QUOTE_YOUPL),
                 {"A": _name_piece(a, rng, alloc), "B": _name_piece(b, rng, alloc),
                  "C": _name_piece(c, rng, alloc),
                  "YouPl": _Piece("你们", rep), "I": _first_piece(a, alloc)})


def _spec_max_mentions(default: int = 8) -> int:
    """单个样本最多能带几个提及切片 —— 必须 <= TaskSpec.max_steps。

    `runtime.GenericTaskDataset` 会做 `spans[:max_steps]` 的**静默截断**：样本里
    提及数超过 max_steps 时，多出来的提及整段丢掉监督，而动作头还被训成「收束」，
    等于教模型「这里没人物」。所以生成侧必须自己把上限卡住，绝不让运行期截断发生。

    上限从同包 `__init__.py` 的 SPEC 惰性读取（那里才是 max_steps 的家）；
    读不到时退回 default（允许本模块被单独 import 做数据体检）。
    """
    try:
        from dtseek.tasks.builtin.person import SPEC  # 惰性导入，避免循环依赖
        return int(SPEC.max_steps)
    except Exception:
        return default


def _mentions(pieces: list[_Piece]) -> int:
    return sum(1 for p in pieces if p.label_pid is not None)


def _make_person_sample(rng: random.Random, n: int, max_len: int,
                        max_mentions: int) -> dict | None:
    """造一条 n 人样本。返回 None 表示这一轮生成失败（由调用方重试）。

    `max_mentions` 是硬上限：一旦加上某子句会超，就跳过该子句（**不是**事后截断切片）。
    """
    if n > max_mentions:
        return None
    persons = _pick_persons(rng, _gender_plan(rng, n))
    if persons is None:
        return None
    alloc = _PidAllocator()
    pieces: list[_Piece] = []

    order = list(persons)
    rng.shuffle(order)
    i = 0
    while i < len(order):
        rem = len(order) - i
        k = (rng.choice([2, 3, 4]) if rem >= 4
             else (rng.choice([2, 3]) if rem >= 3 else rem))
        group, i = order[i:i + k], i + k
        if k == 1:
            pieces += _cl_intro1(group[0], rng, alloc)
        elif k == 2:
            pieces += _cl_intro2(group[0], group[1], rng, alloc)
        elif k == 3:
            pieces += _cl_intro3(group[0], group[1], group[2], rng, alloc)
        else:
            pieces += _cl_intro4(group, rng, alloc)

    males = [p for p in persons if p.gender == "m"]
    females = [p for p in persons if p.gender == "f"]
    uniq = [p for p in (males if len(males) == 1 else []) + (females if len(females) == 1 else [])]
    both_unique = len(males) == 1 and len(females) == 1

    target_len = rng.choice([18, 26, 36, 50, 70, 100, max_len])
    n_follow = rng.randint(1, 4)
    want_quote = n >= 2 and rng.random() < 0.65
    did_quote = False

    def _options() -> list[tuple[str, Callable[[], list[_Piece]]]]:
        """本轮可用的子句候选。**只在构造上不可能歧义时才放进来** —— 歧义句在
        复核器里会被整条丢弃，与其浪费一次尝试，不如一开始就不生成。"""
        opts: list[tuple[str, Callable[[], list[_Piece]]]] = [
            ("attr", lambda: _cl_attr(rng.choice(persons), rng, alloc)),
        ]
        if uniq:
            opts.append(("anaphora", lambda: _cl_anaphora(rng.choice(uniq), rng, alloc)))
        if n < 2:
            return opts
        a, b = rng.sample(persons, 2)
        opts += [
            ("attr2", lambda a=a, b=b: _cl_attr2(a, b, rng, alloc)),
            ("quote_simple", lambda a=a: _cl_quote_simple(a, rng, alloc)),
            ("quote_addr", lambda a=a, b=b: _cl_quote_addr(a, b, rng, alloc)),
        ]
        # 转述/反向指代：句子里两个代词必须**各自**性别唯一 ⇒ 全scene恰好一男一女
        if both_unique and a.gender and b.gender and a.gender != b.gender:
            opts.append(("reverse", lambda a=a, b=b: _cl_reverse(a, b, rng, alloc)))
            if a.gender == "m" and b.gender == "f":
                opts.append(("pair_mix", lambda a=a, b=b: _cl_pair_mix(a, b, rng, alloc)))
        if n >= 3:
            # 刻意不生成复数代词（他们/她们/我们/你们）：它们的答案是**集合**，
            # 一个 span 一个 label 表达不了；标成"集合最小 id"是人为约定，
            # 会让身份指标测到"背没背下约定"而不是"懂不懂指代"。
            # 引语里的第三人称代词：必须是**另一个人**，不能是说话人或受话人自己
            third = [x for x in uniq if x is not a and x is not b]
            if third:
                c = rng.choice(third)
                opts.append(("quote_third", lambda a=a, b=b, c=c: _cl_quote_third(a, b, c, rng, alloc)))
        return opts

    for _ in range(n_follow):
        if len(_pieces_text(pieces)) >= target_len:
            break
        opts = _options()
        if want_quote and not did_quote:
            quotes_only = [o for o in opts if o[0].startswith("quote")]
            if quotes_only:
                opts = quotes_only
        name, fn = rng.choice(opts)
        clause = fn()
        if _mentions(pieces) + _mentions(clause) > max_mentions:
            continue
        if len(_pieces_text(pieces)) + len(_pieces_text(clause)) > max_len:
            continue
        if name.startswith("quote"):
            did_quote = True
        pieces += clause

    if want_quote and not did_quote and n >= 2:
        a, b = rng.sample(persons, 2)
        clause = _cl_quote_addr(a, b, rng, alloc)
        if _mentions(pieces) + _mentions(clause) <= max_mentions and \
           len(_pieces_text(pieces)) + len(_pieces_text(clause)) <= max_len:
            pieces += clause

    text = _pieces_text(pieces)
    if not (4 <= len(text) <= max_len):
        return None
    if any(p.pid == 0 for p in persons):
        return None
    spans = []
    pos = 0
    for pc in pieces:
        if pc.label_pid is not None:
            spans.append({"label": pc.label_pid, "start": pos, "end": pos + len(pc.text),
                          "word": pc.text})
        pos += len(pc.text)
    if len(spans) < 2 or len({p.pid for p in persons}) != n:
        return None
    if len(spans) > max_mentions:
        return None
    truth = _Truth(
        pid_gender={p.pid: p.gender for p in persons},
        pid_surfaces={p.pid: {p.surface} | ({p.alias} if p.alias else set()) for p in persons},
        n_persons=n,
    )
    return {"text": text, "spans": spans, "_truth": truth}


# ── 背景句 ────────────────────────────────────────────────────────────────
#: 手写中性句（零人物词）。
NEUTRAL_SENTENCES = (
    "数据库集群的写入延迟保持在五毫秒以内。",
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
    "气象台发布大风蓝色预警，请注意防范。",
    "该方案经过三轮评审，最终确定了实施路径。",
    "缓存命中率提升后，接口平均耗时下降了四成。",
    "新版固件修复了充电协议兼容性问题。",
    "园区绿化改造工程预计在下月完工。",
    "参考文献列出了近五年该领域的主要进展。",
)

#: 「X说：」结构的负例 —— 有说话动词和冒号，但 X 不是人物。
#: 这是本数据集最容易骗到模型的背景形态（实测：有「说：」就该开火是典型捷径）。
QUOTE_STRUCTURE_NEGATIVES = (
    "俗话说：一日之计在于晨。",
    "书上说：多喝水对身体有好处。",
    "报道说：今年的雨季来得比往年早。",
    "公司说：下个月的团建改到周五。",
    "通知上说：明天上午停水检修。",
    "公告说：图书馆周末延长开放时间。",
    "报告指出：第三季度的出货量有所回升。",
    "新闻里说：那条地铁线年底通车。",
    "说明书上写着：本产品需在阴凉处存放。",
    "文件里说：所有流程都要留档。",
    "研究显示：充足的睡眠有助于记忆力。",
    "规则上写着：每周三下午例行维护。",
    "标签上印着：保质期十二个月。",
    "天气预报说：明天午后有雷阵雨。",
    "老话说：瑞雪兆丰年。",
    "单据上写着：金额已结清。",
)

#: 非人物代词（它/它们）负例：形似代词、但不是人物提及。
NON_PERSON_PRONOUN_NEGATIVES = (
    "这台设备运行得很稳定，它的功耗也很低。",
    "这批零件已经入库，它们明天统一发走。",
    "算法更新了参数，它的收敛速度明显变快。",
    "文件已经归档，它们都放在了共享盘里。",
)


def _mine_background(corpus_glob: str, n: int, max_len: int,
                     max_lines: int | None = None) -> list[str]:
    """从真实语料里挖零人物词的中性句当背景。

    为什么必须挖真实句：手写背景池只有几十句，模型会把「背景长这样」背下来，
    验证集上零误报、真实文本上照开火。背景得像真实分布。
    """
    pool: list[str] = []
    seen: set[str] = set()
    lines = 0
    for path in sorted(glob.glob(corpus_glob)):
        with open(path, encoding="utf-8", errors="ignore") as fp:
            for line in fp:
                lines += 1
                if max_lines is not None and lines > max_lines:
                    return pool
                text = _ROLE_PREFIX.sub("", line.strip())
                for raw in _SENT_SPLIT.findall(text):
                    s = raw.strip()
                    if not _bg_quality_ok(s):
                        continue
                    if _check_background(s, max_len) is not None or s in seen:
                        continue
                    seen.add(s)
                    pool.append(s)
                    if len(pool) >= n:
                        return pool
    return pool


# ══════════════════════════════════════════════════════════════════════════
# 公开构建入口
# ══════════════════════════════════════════════════════════════════════════
#: 人物数 2..7 的配额权重（2 人最常见，7 人稀有但必须有）
PERSON_COUNT_WEIGHTS: dict[int, int] = {2: 30, 3: 24, 4: 18, 5: 12, 6: 9, 7: 7}

LENGTH_BUCKETS = (18, 26, 36, 50, 70, 100, 120)


def _allocate(n_pos: int) -> dict[int, int]:
    """按权重做最大余数分配，再保证 2..7 每个档位至少 1 条（小 target 也不缺席）。"""
    total_w = sum(PERSON_COUNT_WEIGHTS.values())
    raw = {k: n_pos * w / total_w for k, w in PERSON_COUNT_WEIGHTS.items()}
    counts = {k: int(v) for k, v in raw.items()}
    rem = n_pos - sum(counts.values())
    for k in sorted(raw, key=lambda k: -(raw[k] - counts[k]))[:max(rem, 0)]:
        counts[k] += 1
    for k in sorted(counts, key=lambda k: -PERSON_COUNT_WEIGHTS[k]):
        if counts[k] == 0 and n_pos >= len(counts):
            donor = max(counts, key=lambda x: counts[x])
            if counts[donor] > 1:
                counts[donor] -= 1
                counts[k] = 1
    return counts


def build_person_dataset(target_samples: int = 9000, max_len: int = 120,
                         seed: int = 20240927, bg_ratio: float = 0.30,
                         corpus_glob: str = CORPUS_GLOB,
                         max_corpus_lines: int | None = None) -> list[dict]:
    """构建人物追踪数据集。

    Args:
        target_samples: 目标总样本数（默认 9000，规格下限 6000）
        max_len: 文本长度上界（默认 120 字，够装多轮对话；下界固定 4）
        seed: 随机种子；默认值是历史固定值，改动它会改变数据集内容
        bg_ratio: 背景样本（零人物提及）配额
        corpus_glob: 真实对话语料（只用来挖背景句）
        max_corpus_lines: 扫描语料的硬上界（调试/测试用，None = 扫到够为止）
    """
    audit = audit_templates()
    if audit["person_noun_leaks"]:
        bad = audit["person_noun_leaks"][:5]
        raise ValueError(f"模板字面混入人物名词（会漏标）: {bad}")

    rng = random.Random(seed)
    n_bg = int(target_samples * bg_ratio)
    n_pos = target_samples - n_bg
    max_mentions = _spec_max_mentions()
    counts = _allocate(n_pos)
    for k in [k for k in counts if k > max_mentions]:
        print(f"  ⚠ {k} 人档位超过 TaskSpec.max_steps={max_mentions}，已跳过（运行期会截断监督）")
        counts.pop(k)
    print(f"  单样本提及上限: {max_mentions}（= TaskSpec.max_steps，生成侧卡死，运行期不截断）")

    dataset: list[dict] = []
    drops: Counter = Counter()
    used_text: Counter = Counter()
    per_count: Counter = Counter()

    for n in sorted(counts):
        need = counts[n]
        made = 0
        attempts = 0
        max_attempts = need * 60 + 3000
        while made < need and attempts < max_attempts:
            attempts += 1
            sample = _make_person_sample(rng, n, max_len, max_mentions)
            if sample is None:
                drops["生成失败"] += 1
                continue
            reason = _check_sample(sample["text"], sample["spans"], sample["_truth"], max_len)
            if reason is not None:
                drops[reason] += 1
                continue
            if used_text[sample["text"]] >= 2:
                drops["重复文本"] += 1
                continue
            used_text[sample["text"]] += 1
            dataset.append({"text": sample["text"], "spans": sample["spans"]})
            per_count[n] += 1
            made += 1
        if made < need:
            print(f"  ⚠ {n} 人档位只造出 {made}/{need}（尝试 {attempts} 次）")

    # 背景：手写中性 + 「X说：」结构负例 + 非人物代词负例 + 真实语料挖掘
    hand = list(NEUTRAL_SENTENCES + QUOTE_STRUCTURE_NEGATIVES + NON_PERSON_PRONOUN_NEGATIVES)
    n_mine = max(int(n_bg * 1.5), 400)
    mined = _mine_background(corpus_glob, n_mine, max_len, max_corpus_lines)
    bg_pool: list[tuple[str, str]] = [("synthetic", s) for s in hand]
    bg_pool += [("real", s) for s in mined]
    good_bg = [(src, s) for src, s in bg_pool if _check_background(s, max_len) is None]
    if len(good_bg) < 60:
        raise ValueError(f"可用背景句只有 {len(good_bg)} 条，背景太少模型会学成永远开火")
    print(f"  背景句池：手写 {len(hand)} 条 + 真实语料挖到 {len(mined)} 条 "
          f"（可用 {len(good_bg)} 条）")

    bg_used: Counter = Counter()
    n_real_bg = 0
    made_bg = 0
    for _ in range(n_bg * 60):
        if made_bg >= n_bg:
            break
        src, s = rng.choice(good_bg)
        if bg_used[s] >= 3:
            continue
        bg_used[s] += 1
        dataset.append({"text": s, "spans": []})
        n_real_bg += int(src == "real")
        made_bg += 1
    if made_bg < n_bg:
        print(f"  ⚠ 背景句只填到 {made_bg}/{n_bg}")

    rng.shuffle(dataset)
    _report(dataset, drops, per_count, n_real_bg)
    return dataset


def _report(dataset: list[dict], drops: Counter, per_count: Counter,
            n_real_bg: int) -> None:
    n = len(dataset)
    n_bg = sum(1 for d in dataset if not d["spans"])
    dist = Counter(len({s["label"] for s in d["spans"]}) for d in dataset)
    total_spans = sum(len(d["spans"]) for d in dataset)
    n_pos_samples = n - n_bg
    n_spans_pos = total_spans
    n_quote = sum(1 for d in dataset if "：“" in d["text"] or "”" in d["text"])
    n_plural = sum(1 for d in dataset
                   for s in d["spans"] if s["word"] in _PLURAL_PRONOUNS)
    n_alias = sum(1 for d in dataset
                  for s in d["spans"] if _PERSON_VOCAB.get(s["word"]) == "alias")
    print(f"  总样本数: {n}")
    print(f"  人物数分布（去重 label 数）: "
          + " | ".join(f"{k}人:{dist.get(k, 0)}" for k in range(0, 8)))
    print(f"  背景占比: {n_bg}/{n} = {n_bg / max(n, 1) * 100:.1f}%")
    print(f"  平均提及数: {total_spans / max(n, 1):.2f}（仅正例 {n_spans_pos / max(n_pos_samples, 1):.2f}）")
    print(f"  真实/合成比例: 真实(挖掘背景) {n_real_bg} | "
          f"合成 {n - n_real_bg}（人物切片 100% 合成）")
    print(f"  直接引语样本: {n_quote} | 复数代词提及: {n_plural} | 别名提及: {n_alias}")
    if drops:
        print("  丢弃统计: " + " | ".join(f"{k}:{v}" for k, v in drops.most_common()))


def sample_stats(dataset: list[dict]) -> dict:
    """给探针/报告用的结构化统计。"""
    dist = Counter(len({s["label"] for s in d["spans"]}) for d in dataset)
    return {
        "n": len(dataset),
        "person_count_dist": {k: dist.get(k, 0) for k in range(0, 8)},
        "bg": dist.get(0, 0),
        "bg_ratio": dist.get(0, 0) / max(len(dataset), 1),
        "avg_spans": sum(len(d["spans"]) for d in dataset) / max(len(dataset), 1),
        "quotable": sum(1 for d in dataset if "：“" in d["text"]),
    }


# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":  # pragma: no cover
    ds = build_person_dataset(target_samples=600)
    quotes = [d for d in ds if "：“" in d["text"]]
    bgs = [d for d in ds if not d["spans"]]
    multi = [d for d in ds if len({s["label"] for s in d["spans"]}) >= 3]
    picks: list[dict] = []
    seen_text: set[str] = set()

    def _add(pool: list[dict], k: int) -> None:
        n = 0
        for d in pool:
            if n >= k:
                break
            if d["text"] in seen_text:
                continue
            seen_text.add(d["text"])
            picks.append(d)
            n += 1

    _add(quotes, 3)
    _add(bgs, 2)
    _add(multi, 1)
    for d in ds:
        if len(picks) >= 12:
            break
        if d["text"] not in seen_text:
            seen_text.add(d["text"])
            picks.append(d)
    for d in picks[:12]:
        kind = "背景" if not d["spans"] else f"{len({s['label'] for s in d['spans']})}人"
        print(f"\n[{kind}] {d['text']}")
        for s in d["spans"]:
            print(f"    {s['word']!r} label={s['label']} [{s['start']},{s['end']})")
    print("\n" + str(sample_stats(ds)))
