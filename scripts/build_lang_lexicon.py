"""从开源成语词典 + 真实语料生成 `src/dtseek/tasks/builtin/relation/lexicon.py`（小学语文成语近义词 / 反义词词表）。

来源只留 `idiom/idiom.json`（成语级 similar/opposite 逐条编纂，质量可用）；
`word/word.json` 的词级 similar/opposite 抽样有错（自然~自由、条件~要求），本脚本不再使用。

流程（每一步都打印进度与淘汰数）：
    1. 容错加载 idiom.json / char_base.json
    2. 抽出 similar -> 近义候选、opposite -> 反义候选（无序去重，每对只留一个方向）
    3. 逐条硬性过滤：4 字纯中文、字频<=2、无包含、共享字符数<=1，再让反义表剔除近义候选
    4. 用 nanoSeek 中文对话语料统计候选词出现次数（Aho-Corasick 一次扫全语料，结果缓存到 /tmp）
    5. 打印 200/100/50/20/10/5/3/2/1 各门槛下剩余对数，按选定门槛 + 上限 + 一词只用一次贪心选对
    6. 写出数据模块，内含 PROVENANCE / SYNONYM_PAIRS / ANTONYM_PAIRS / validate()

用法：
    uv run python scripts/build_lang_lexicon.py            # 用默认路径
    uv run python scripts/build_lang_lexicon.py --no-cache # 强制重扫语料
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# 默认输入 / 输出
# ---------------------------------------------------------------------------
DICT_DIR = Path("/tmp/chinese-dictionary")
CORPUS_DIR = Path("/home/vesita/coding/my/nanoSeek/data/chinese")
CORPUS_GLOB = "*dialogue.txt"
OUT_PATH = Path(__file__).resolve().parent.parent / "src" / "dtseek" / "tasks" / "builtin" / "relation" / "lexicon.py"
COUNT_CACHE = Path("/tmp/dtseek_lang_lexicon_counts.json")

SOURCE_REPO = "https://github.com/mapull/chinese-dictionary"
SOURCE_LICENSE = "MIT"
SOURCE_COMMIT = "e804ada"

# ---------------------------------------------------------------------------
# 硬性阈值（写进 PROVENANCE）
# ---------------------------------------------------------------------------
WORD_LEN = 4  # 只收四字成语
MAX_CHAR_FREQUENCY = 2  # 只收 3500 常用字（词典 frequency 0..2）
MAX_SHARED_CHARS = 1  # 两词共享字符数（多重集交集大小）<= 1
# 语料门槛：min(count_a, count_b) 必须 >= MIN_COUNT。
# 成语在对话语料里的出现次数远低于常用词，门槛扫描后选定（理由见 MIN_COUNT_RATIONALE）。
MIN_COUNT = 50
MIN_COUNT_RATIONALE = ""
MAX_SYNONYM_PAIRS = 400
MAX_ANTONYM_PAIRS = 350

THRESHOLD_SWEEP = (200, 100, 50, 20, 10, 5, 3, 2, 1)

CJK_RE = re.compile(r"^[\u4e00-\u9fff]{" + str(WORD_LEN) + r"}$")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 1. 容错加载
# ---------------------------------------------------------------------------
def load_jsonlish(path: Path) -> list[dict]:
    """词典文件是「一行一个 JSON 对象」的 JSONL 风格（可能带数组括号），两路都试。"""
    text = path.read_text(encoding="utf-8")
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, list) else [obj]
    except json.JSONDecodeError:
        rows: list[dict] = []
        bad = 0
        for line in text.splitlines():
            s = line.strip()
            if not s or s in ("[", "]"):
                continue
            s = s.removesuffix(",")
            try:
                rows.append(json.loads(s))
            except json.JSONDecodeError:
                bad += 1
        if bad:
            log(f"  警告：{path.name} 有 {bad} 行无法解析，已跳过")
        return rows


# ---------------------------------------------------------------------------
# 3. 过滤谓词
# ---------------------------------------------------------------------------
def shared_char_count(a: str, b: str) -> int:
    """两词共享字符数：按多重集求交（重复字符按出现次数计），如 一心一意/一意孤行 = 3。"""
    return sum((Counter(a) & Counter(b)).values())


def make_filters(char_freq: dict[str, int]):
    """返回按顺序执行的过滤步骤：[(名字, 谓词)]。"""

    def rule1(a: str, b: str) -> bool:
        return bool(CJK_RE.match(a)) and bool(CJK_RE.match(b))

    def rule2(a: str, b: str) -> bool:
        return all(char_freq.get(c, 99) <= MAX_CHAR_FREQUENCY for c in a + b)

    def rule3(a: str, b: str) -> bool:
        return a not in b and b not in a

    def rule4(a: str, b: str) -> bool:
        return shared_char_count(a, b) <= MAX_SHARED_CHARS

    return [
        (f"规则1 词长恒为 {WORD_LEN} 且纯中文", rule1),
        (f"规则2 每字 frequency<={MAX_CHAR_FREQUENCY}", rule2),
        ("规则3 两词不互为子串（无包含）", rule3),
        (f"规则4 两词共享字符数<={MAX_SHARED_CHARS}（按多重集求交）", rule4),
    ]


# ---------------------------------------------------------------------------
# 4. 语料词频统计
# ---------------------------------------------------------------------------
def build_matcher(words: list[str]):
    """优先用 pyahocorasick（C 实现，2 亿字约 80 秒）；否则退化为纯 Python trie。"""
    try:
        import ahocorasick  # type: ignore

        automaton = ahocorasick.Automaton()
        for idx, w in enumerate(words):
            automaton.add_word(w, idx)
        automaton.make_automaton()
        return "ahocorasick", automaton
    except ImportError:
        root: dict = {}
        for idx, w in enumerate(words):
            node = root
            for ch in w:
                node = node.setdefault(ch, {})
            node["#"] = idx
        return "trie", root


def count_in_text(text: str, engine: str, matcher, counts: list[int]) -> None:
    if engine == "ahocorasick":
        for _end, idx in matcher.iter(text):
            counts[idx] += 1
        return
    root = matcher
    n = len(text)
    for i in range(n):
        node = root.get(text[i])
        if node is None:
            continue
        j = i + 1
        while node is not None:
            hit = node.get("#")
            if hit is not None:
                counts[hit] += 1
            if j >= n:
                break
            node = node.get(text[j])
            j += 1


def corpus_word_counts(words: list[str], corpus_dir: Path, pattern: str, use_cache: bool) -> tuple[dict[str, int], int]:
    files = sorted(corpus_dir.glob(pattern))
    if not files:
        raise SystemExit(f"语料目录 {corpus_dir} 下没有匹配 {pattern} 的文件")

    if use_cache and COUNT_CACHE.exists():
        cached = json.loads(COUNT_CACHE.read_text(encoding="utf-8"))
        if set(words) <= set(cached.get("counts", {})):
            log(f"  命中缓存 {COUNT_CACHE}（{len(cached['counts'])} 词），跳过语料扫描")
            return {w: cached["counts"].get(w, 0) for w in words}, int(cached.get("corpus_chars", 0))
        log("  缓存不完整，重新扫描语料")

    engine, matcher = build_matcher(words)
    counts = [0] * len(words)
    log(f"  匹配引擎={engine}，候选词 {len(words)} 个，语料 {len(files)} 个文件")
    total_chars = 0
    t0 = time.time()
    for path in files:
        t1 = time.time()
        text = path.read_text(encoding="utf-8", errors="ignore")
        total_chars += len(text)
        count_in_text(text, engine, matcher, counts)
        log(f"    {path.name}: {len(text):,} 字，本轮 {time.time() - t1:.1f}s，累计 {time.time() - t0:.1f}s")
    result = {w: counts[i] for i, w in enumerate(words)}
    COUNT_CACHE.write_text(
        json.dumps(
            {"corpus_files": [p.name for p in files], "corpus_chars": total_chars, "counts": result},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    log(f"  语料共 {total_chars:,} 字，统计完成并缓存到 {COUNT_CACHE}")
    return result, total_chars


# ---------------------------------------------------------------------------
# 5. 贪心选对
# ---------------------------------------------------------------------------
def pair_min_count(counts: dict[str, int], pair: tuple[str, str]) -> int:
    return min(counts.get(pair[0], 0), counts.get(pair[1], 0))


def select_pairs(
    ranked: dict[str, list[tuple[str, str]]],
    counts: dict[str, int],
    min_count: int,
    cap_synonym: int,
    cap_antonym: int,
) -> tuple[dict[str, list[tuple[str, str]]], dict[str, tuple[int, int]]]:
    """近义先选、反义后选，共享同一个已用词集合；返回选中对与 (低于门槛, 词被占用) 淘汰数。"""
    used: set[str] = set()
    selected: dict[str, list[tuple[str, str]]] = {}
    stats: dict[str, tuple[int, int]] = {}
    for table, cap in (("synonym", cap_synonym), ("antonym", cap_antonym)):
        below = collide = 0
        kept: list[tuple[str, str]] = []
        for pair in ranked[table]:
            if pair_min_count(counts, pair) < min_count:
                below += 1
                continue
            if pair[0] in used or pair[1] in used:
                collide += 1
                continue
            used.update(pair)
            kept.append(pair)
            if len(kept) >= cap:
                break
        selected[table] = kept
        stats[table] = (below, collide)
    return selected, stats


# ---------------------------------------------------------------------------
# 6. 生成模块
# ---------------------------------------------------------------------------
def format_pairs(pairs: list[tuple[str, str]], per_line: int = 4) -> str:
    lines = []
    for i in range(0, len(pairs), per_line):
        chunk = ", ".join(f'("{a}", "{b}")' for a, b in pairs[i : i + per_line])
        lines.append(f"    {chunk},")
    return "\n".join(lines)


MODULE_TEMPLATE = '''"""小学语文成语近义词 / 反义词词表（由 scripts/build_lang_lexicon.py 从开源词典生成）。

来源：https://github.com/mapull/chinese-dictionary (MIT)，commit e804ada
生成方式：见 PROVENANCE；不要手改本文件，改生成脚本后重跑。
"""
from __future__ import annotations

import re
from collections import Counter

PROVENANCE: dict = __PROVENANCE__

# char_base.frequency <= __MAX_CHAR_FREQUENCY__ 的全部汉字（__N_COMMON_CHARS__ 个），validate() 用它做「常用字」兜底检查。
_COMMON_CHARS: str = (
__CHAR_BLOCK__
)
_COMMON_CHAR_SET: frozenset[str] = frozenset(_COMMON_CHARS)

_WORD_LEN = __WORD_LEN__
_MAX_SHARED_CHARS = __MAX_SHARED_CHARS__
_CJK_RE = re.compile(r"^[\\u4e00-\\u9fff]{" + str(_WORD_LEN) + r"}$")

# 近义词对（按 min(语料词频) 降序）
SYNONYM_PAIRS: list[tuple[str, str]] = [
__SYNONYM_BODY__
]

# 反义词对（按 min(语料词频) 降序）
ANTONYM_PAIRS: list[tuple[str, str]] = [
__ANTONYM_BODY__
]

_MIN_PAIRS_PER_TABLE = 100


def _shared_char_count(a: str, b: str) -> int:
    """两词共享字符数：按多重集求交（重复字符按出现次数计）。"""
    return sum((Counter(a) & Counter(b)).values())


def validate() -> list[str]:
    """返回全部违规项；空列表 = 通过。

    逐条检查生成时的硬性规则：每词长度恒为 4 且纯中文、常用字、无包含、
    两词共享字符数 <= 1、同表无重复对（含逆序）、两表词集合零交集、
    一个词在两表里合计只出现 1 次、反义对不落近义表、两表各不少于 100 对。
    """
    problems: list[str] = []

    if not SYNONYM_PAIRS:
        problems.append("SYNONYM_PAIRS 为空")
    elif len(SYNONYM_PAIRS) < _MIN_PAIRS_PER_TABLE:
        problems.append(f"SYNONYM_PAIRS 只有 {len(SYNONYM_PAIRS)} 对，少于 {_MIN_PAIRS_PER_TABLE} 对")
    if not ANTONYM_PAIRS:
        problems.append("ANTONYM_PAIRS 为空")
    elif len(ANTONYM_PAIRS) < _MIN_PAIRS_PER_TABLE:
        problems.append(f"ANTONYM_PAIRS 只有 {len(ANTONYM_PAIRS)} 对，少于 {_MIN_PAIRS_PER_TABLE} 对")

    tables = (("SYNONYM", SYNONYM_PAIRS), ("ANTONYM", ANTONYM_PAIRS))
    syn_set = set(SYNONYM_PAIRS) | {(b, a) for a, b in SYNONYM_PAIRS}
    seen_pairs: dict[tuple[str, str], str] = {}

    for table, pairs in tables:
        for a, b in pairs:
            tag = f"{table}({a},{b})"
            for w in (a, b):
                if len(w) != _WORD_LEN or not _CJK_RE.match(w):
                    problems.append(f"规则1 违规 {tag}：{w!r} 不是 {_WORD_LEN} 字纯中文")
                for ch in w:
                    if ch not in _COMMON_CHAR_SET:
                        problems.append(f"规则2 违规 {tag}：{w!r} 含非常用字 {ch!r}")
            if a == b:
                problems.append(f"规则3 违规 {tag}：两词相同")
            if a in b or b in a:
                problems.append(f"规则3 违规 {tag}：互为子串")
            shared = _shared_char_count(a, b)
            if shared > _MAX_SHARED_CHARS:
                problems.append(f"规则4 违规 {tag}：两词共享字符数 {shared} > {_MAX_SHARED_CHARS}")
            if (a, b) in seen_pairs:
                problems.append(f"规则5 违规 {tag}：与 {seen_pairs[(a, b)]} 重复")
            elif (b, a) in seen_pairs:
                problems.append(f"规则5 违规 {tag}：与 {seen_pairs[(b, a)]} 互为逆序")
            else:
                seen_pairs[(a, b)] = tag
            if table == "ANTONYM" and (a, b) in syn_set:
                problems.append(f"规则7 违规 {tag}：同一对同时出现在近义表与反义表")

    word_counts = Counter(w for _t, pairs in tables for pair in pairs for w in pair)
    for w, n in sorted(word_counts.items()):
        if n > 1:
            problems.append(f"规则6 违规：词 {w!r} 在两张表里合计出现 {n} 次（应 <= 1）")

    syn_words = {w for pair in SYNONYM_PAIRS for w in pair}
    ant_words = {w for pair in ANTONYM_PAIRS for w in pair}
    overlap = syn_words & ant_words
    if overlap:
        problems.append(f"两表词集合交集非空：{sorted(overlap)[:10]}")
    return problems
'''


def render_module(
    synonyms: list[tuple[str, str]],
    antonyms: list[tuple[str, str]],
    common_chars: str,
    corpus_files: list[str],
    corpus_chars: int,
    threshold_sweep: dict[str, list[int]],
    filter_report: dict[str, list[list[int]]],
    min_count_rationale: str,
) -> str:
    char_block = "\n".join(f'    "{common_chars[i : i + 60]}"' for i in range(0, len(common_chars), 60))

    provenance = {
        "source_repo": SOURCE_REPO,
        "source_commit": SOURCE_COMMIT,
        "source_license": SOURCE_LICENSE,
        "source_files": ["idiom/idiom.json", "character/char_base.json"],
        "source_fields": {"synonym": ["idiom.similar"], "antonym": ["idiom.opposite"]},
        "excluded_source": "word/word.json 未使用：词级 similar/opposite 抽样有错（自然~自由、条件~要求）",
        "corpus_files": corpus_files,
        "corpus_chars": corpus_chars,
        "filters": [
            f"词条长度恒为 {WORD_LEN} 字且完全由 \\u4e00-\\u9fff 组成（无标点/字母/数字）",
            f"词条每个字的 char_base.frequency <= {MAX_CHAR_FREQUENCY}（3500 常用字）",
            "两词不互为子串（禁止包含关系）",
            f"两词共享字符数 <= {MAX_SHARED_CHARS}（按多重集求交，含重复字符计数，如 通邑大都/通都大邑 共享 3）",
            "近义候选对不再进入反义表（含逆序）",
            "一个词在近义表与反义表中合计只出现 1 次",
            "每对只存一次、不存逆序（无序去重，键为字典序规范化后的二元组）",
            "先选近义后选反义，共享同一个已用词集合",
            "剔除词典自相矛盾的对：同一对既被标成近义又被标成反义（如 正人君子/跳梁小丑）",
        ],
        "thresholds": {
            "min_count": MIN_COUNT,
            "max_synonym_pairs": MAX_SYNONYM_PAIRS,
            "max_antonym_pairs": MAX_ANTONYM_PAIRS,
            "word_len": WORD_LEN,
            "max_char_frequency": MAX_CHAR_FREQUENCY,
            "max_shared_chars": MAX_SHARED_CHARS,
        },
        "min_count_definition": "min(语料中 a 的出现次数, 语料中 b 的出现次数)；出现次数为子串匹配计数",
        "min_count_rationale": min_count_rationale,
        "threshold_sweep": threshold_sweep,
        "threshold_sweep_columns": ["近义达标", "近义选后", "反义达标", "反义选后"],
        "filter_report": filter_report,
        "generated_pairs": {"synonym": len(synonyms), "antonym": len(antonyms)},
    }
    prov_lines = "\n".join(f"    {key!r}: {value!r}," for key, value in provenance.items())

    source = MODULE_TEMPLATE
    source = source.replace("__PROVENANCE__", "{\n" + prov_lines + "\n}")
    source = source.replace("__MAX_CHAR_FREQUENCY__", str(MAX_CHAR_FREQUENCY))
    source = source.replace("__N_COMMON_CHARS__", str(len(common_chars)))
    source = source.replace("__CHAR_BLOCK__", char_block)
    source = source.replace("__WORD_LEN__", str(WORD_LEN))
    source = source.replace("__MAX_SHARED_CHARS__", str(MAX_SHARED_CHARS))
    source = source.replace("__SYNONYM_BODY__", format_pairs(synonyms))
    source = source.replace("__ANTONYM_BODY__", format_pairs(antonyms))
    return source


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="生成小学语文成语近义/反义词表")
    parser.add_argument("--dict-dir", type=Path, default=DICT_DIR)
    parser.add_argument("--corpus-dir", type=Path, default=CORPUS_DIR)
    parser.add_argument("--corpus-glob", default=CORPUS_GLOB)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument("--no-cache", action="store_true", help="忽略 /tmp 词频缓存，强制重扫语料")
    parser.add_argument("--min-count", type=int, default=None, help="临时覆盖 MIN_COUNT（不改常量）")
    args = parser.parse_args()
    min_count = MIN_COUNT if args.min_count is None else args.min_count
    min_count_rationale = MIN_COUNT_RATIONALE

    log("步骤 1/6 加载词典（只用 idiom.json + char_base.json）")
    idiom_rows = load_jsonlish(args.dict_dir / "idiom" / "idiom.json")
    char_rows = load_jsonlish(args.dict_dir / "character" / "char_base.json")
    char_freq = {r["char"]: int(r["frequency"]) for r in char_rows if "char" in r and "frequency" in r}
    log(f"  成语 {len(idiom_rows)} 条，汉字 {len(char_freq)} 个（word/word.json 已弃用）")

    log("步骤 2/6 抽取 similar / opposite 原始对（无序去重）")
    raw: dict[str, dict[tuple[str, str], int]] = {"synonym": {}, "antonym": {}}
    field_of = {"synonym": "similar", "antonym": "opposite"}
    for table, field in field_of.items():
        entries = 0
        for row in idiom_rows:
            a = row.get("word")
            if not isinstance(a, str):
                continue
            for b in row.get(field) or []:
                if not isinstance(b, str) or a == b:
                    continue
                entries += 1
                key = (a, b) if a <= b else (b, a)
                raw[table][key] = raw[table].get(key, 0) + 1
        log(f"  {table}: 有效条目 {entries} 条 -> 无序唯一对 {len(raw[table])} 条")

    log("步骤 3/6 逐条硬性过滤")
    steps = make_filters(char_freq)
    filtered: dict[str, list[tuple[str, str]]] = {}
    filter_report: dict[str, list[list[int]]] = {}
    for table, pairs in raw.items():
        current = list(pairs.items())
        log(f"  [{table}] 起点 {len(current)} 对")
        report: list[list[int]] = []
        for name, predicate in steps:
            kept = [item for item in current if predicate(*item[0])]
            report.append([len(current), len(kept), len(current) - len(kept)])
            log(f"  [{table}] {name}: {len(current)} -> {len(kept)}（淘汰 {len(current) - len(kept)}）")
            current = kept
        filtered[table] = [pair for pair, _ in current]
        filter_report[table] = report

    # 词典自相矛盾：同一对在某条目标成近义、在另一条目标成反义 —— 数据本身打架，直接剔除。
    # 实测抓到 `正人君子/跳梁小丑`（近义表方向就是错的），与人工复核结论一致。
    contradict = set(raw["synonym"]) & set(raw["antonym"])
    for table in ("synonym", "antonym"):
        before = len(filtered[table])
        filtered[table] = [p for p in filtered[table] if p not in contradict]
        dropped = before - len(filtered[table])
        filter_report[table].append([before, len(filtered[table]), dropped])
        log(f"  [{table}] 规则6 剔除词典自相矛盾对（近义/反义都标过）: {before} -> "
            f"{len(filtered[table])}（淘汰 {dropped}）")

    # 反义表剔除近义候选（含逆序；键已规范化，直接比较二元组）
    syn_keys = set(filtered["synonym"])
    before = len(filtered["antonym"])
    filtered["antonym"] = [p for p in filtered["antonym"] if p not in syn_keys]
    dropped = before - len(filtered["antonym"])
    filter_report["antonym"].append([before, len(filtered["antonym"]), dropped])
    log(f"  [antonym] 规则5 剔除与近义候选重合的对（含逆序）: {before} -> {len(filtered['antonym'])}（淘汰 {dropped}）")

    log("步骤 4/6 统计语料词频")
    candidates = sorted({w for pairs in filtered.values() for pair in pairs for w in pair})
    log(f"  待统计候选词 {len(candidates)} 个")
    counts, corpus_chars = corpus_word_counts(candidates, args.corpus_dir, args.corpus_glob, not args.no_cache)

    log("步骤 5/6 门槛扫描 + 贪心选对（min 词频降序，阈值 + 上限 + 一词只用一次）")
    ranked = {table: sorted(pairs, key=lambda p: (-pair_min_count(counts, p), p)) for table, pairs in filtered.items()}
    log("  门槛扫描（选后 = 共享已用词集合贪心后的对数）")
    print(f"    {'门槛':>6} {'近义达标':>8} {'近义选后':>8} {'反义达标':>8} {'反义选后':>8}", flush=True)
    threshold_sweep: dict[str, list[int]] = {}
    for th in THRESHOLD_SWEEP:
        sel, _ = select_pairs(ranked, counts, th, MAX_SYNONYM_PAIRS, MAX_ANTONYM_PAIRS)
        syn_over = sum(1 for p in ranked["synonym"] if pair_min_count(counts, p) >= th)
        ant_over = sum(1 for p in ranked["antonym"] if pair_min_count(counts, p) >= th)
        threshold_sweep[str(th)] = [syn_over, len(sel["synonym"]), ant_over, len(sel["antonym"])]
        print(f"    {th:>6} {syn_over:>8} {len(sel['synonym']):>8} {ant_over:>8} {len(sel['antonym']):>8}", flush=True)

    if str(min_count) in threshold_sweep:
        syn_sel = threshold_sweep[str(min_count)][1]
        ant_sel = threshold_sweep[str(min_count)][3]
        # 门槛是「精度 vs 数量」的取舍，不是凑数：反义表低词频尾部实测是主要噪声源
        # （30-40 区间里 拐弯抹角/短兵相接、九牛一毛/雨后春笋 这类完全无关的对最集中），
        # 所以宁可少收也要把尾部切掉。
        min_count_rationale = (
            "成语在语料里的出现次数远低于常用词，原 500 门槛会把候选几乎清空。"
            f"门槛扫描 {list(THRESHOLD_SWEEP)} 显示 MIN_COUNT={min_count} 时近义剩 {syn_sel} 对、"
            f"反义剩 {ant_sel} 对。"
            "取 50 而不是更低的 20（20 时反义 350 对）：成语词典 opposite 字段在低词频尾部明显更不可靠，"
            "实测 30-40 词频区间集中了 拐弯抹角/短兵相接、九牛一毛/雨后春笋 这类完全无关的对。"
            "把反义表下限抬到 50 会从 350 降到 138 对，但换来的标签精度更值 —— "
            "近义表在 50 与 20 下选出的都是按词频降序的前 400 对，完全一致，所以这次调整只影响反义表。"
        )
    log(f"  选定门槛 MIN_COUNT={min_count}")

    selected, stats = select_pairs(ranked, counts, min_count, MAX_SYNONYM_PAIRS, MAX_ANTONYM_PAIRS)
    for table in ("synonym", "antonym"):
        below, collide = stats[table]
        log(
            f"  [{table}] 候选 {len(ranked[table])} 对：低于词频门槛淘汰 {below}，词已被占用淘汰 {collide}，"
            f"保留 {len(selected[table])} 对"
        )
        filter_report[table].append([len(ranked[table]), len(selected[table]), len(ranked[table]) - len(selected[table])])

    log("步骤 6/6 写出数据模块")
    common_chars = "".join(sorted(c for c, f in char_freq.items() if f <= MAX_CHAR_FREQUENCY))
    source = render_module(
        selected["synonym"],
        selected["antonym"],
        common_chars,
        [p.name for p in sorted(args.corpus_dir.glob(args.corpus_glob))],
        corpus_chars,
        threshold_sweep,
        filter_report,
        min_count_rationale,
    )
    args.out.write_text(source, encoding="utf-8")
    log(f"  已写出 {args.out}（{len(source.splitlines())} 行）")

    min_counts = {p: pair_min_count(counts, p) for t in selected for p in selected[t]}

    def show(title: str, pairs: list[tuple[str, str]], head: bool) -> None:
        ordered = sorted(pairs, key=lambda p: (-min_counts[p], p))
        picked = ordered[:20] if head else ordered[-20:]
        log(f"  {title}")
        for a, b in picked:
            print(f"      {a} ~ {b}  min_count={min_counts[(a, b)]}")

    log(f"最终：近义 {len(selected['synonym'])} 对，反义 {len(selected['antonym'])} 对")
    show("最常用 20 对（近义）", selected["synonym"], True)
    show("最不常用 20 对（近义）", selected["synonym"], False)
    show("最常用 20 对（反义）", selected["antonym"], True)
    show("最不常用 20 对（反义）", selected["antonym"], False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
