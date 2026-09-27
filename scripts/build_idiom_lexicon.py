"""生成 src/dtseek/tasks/builtin/idiom/lexicon.py：成语识别任务卡用的常用成语白名单。

数据源：
  - /tmp/chinese-dictionary/idiom/idiom.json      开源成语词典（49639 条）
  - /tmp/chinese-dictionary/character/char_base.json  汉字字频表（21056 字，frequency 0 最常用 … 5 生僻）
  - /home/vesita/coding/my/nanoSeek/data/chinese/*dialogue.txt  约 2 亿字真实中文语料

淘汰顺序与理由见生成的 PROVENANCE。语料计数缓存在 /tmp/dtseek_idiom_counts.json，
按「语料文件列表 + 各文件字节数」校验，命中即不重扫。

用法：
  uv run python scripts/build_idiom_lexicon.py            # 生成 + 落盘
  uv run python scripts/build_idiom_lexicon.py --dry-run  # 只打印各步淘汰数
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
IDIOM_JSON = Path("/tmp/chinese-dictionary/idiom/idiom.json")
CHAR_BASE_JSON = Path("/tmp/chinese-dictionary/character/char_base.json")
CORPUS_DIR = Path("/home/vesita/coding/my/nanoSeek/data/chinese")
CORPUS_GLOB = "*dialogue.txt"
CACHE_PATH = Path("/tmp/dtseek_idiom_counts.json")
OUT_PATH = REPO_ROOT / "src" / "dtseek" / "tasks" / "builtin" / "idiom" / "lexicon.py"

SOURCE_REPO = "https://github.com/mapull/chinese-dictionary"
SOURCE_COMMIT = "e804ada"
SOURCE_LICENSE = "MIT"

WORD_LEN = 4
MAX_CHAR_FREQUENCY = 2
THRESHOLD_CANDIDATES = (500, 200, 100, 50, 20, 10, 5)
FINAL_COUNT_RANGE = (1500, 4000)
MAX_FINAL_COUNT = 4000

HAN_RE = re.compile(r"^[\u4e00-\u9fff]+$")


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- 读取输入


def _load_records(path: Path) -> list[dict]:
    """容错加载：先整体 json.loads，失败再按行解析（去掉尾逗号与数组括号）。"""
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    records = []
    for line in text.splitlines():
        line = line.strip().rstrip(",")
        if not line or line in ("[", "]"):
            continue
        records.append(json.loads(line))
    return records


def load_idioms(path: Path) -> list[dict]:
    records = _load_records(path)
    log(f"[加载] 成语词典 {path.name}: {len(records)} 条")
    return records


def load_char_frequency(path: Path) -> dict[str, int]:
    records = _load_records(path)
    freq = {rec["char"]: rec["frequency"] for rec in records if "char" in rec and "frequency" in rec}
    log(f"[加载] 字频表 {path.name}: {len(freq)} 字，其中 frequency<={MAX_CHAR_FREQUENCY} 的 {sum(1 for f in freq.values() if f <= MAX_CHAR_FREQUENCY)} 字")
    return freq


# --------------------------------------------------------------------------- 语料计数


def corpus_files() -> list[Path]:
    return sorted(CORPUS_DIR.glob(CORPUS_GLOB))


def corpus_signature(files: list[Path]) -> list[list]:
    return [[f.name, f.stat().st_size] for f in files]


def _scan_ahocorasick(words: list[str], files: list[Path]) -> Counter:
    import ahocorasick

    automaton = ahocorasick.Automaton()
    for idx, word in enumerate(words):
        automaton.add_word(word, idx)
    automaton.make_automaton()
    counts: Counter = Counter()
    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for _end, idx in automaton.iter(text):
            counts[idx] += 1
    return counts


def _scan_pure_python(words: list[str], files: list[Path]) -> Counter:
    """纯 Python 兜底：滑动窗口取 4 字串，命中候选集合才计数（等长模式与 AC 等价）。"""
    wanted = set(words)
    index = {word: idx for idx, word in enumerate(words)}
    counts: Counter = Counter()
    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for i in range(len(text) - WORD_LEN + 1):
            window = text[i : i + WORD_LEN]
            if window in wanted:
                counts[index[window]] += 1
    return counts


def read_cache(signature: list[list]) -> dict | None:
    if not CACHE_PATH.exists():
        return None
    try:
        cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if cache.get("corpus_signature") != signature:
        return None
    if not isinstance(cache.get("counts"), dict):
        return None
    return cache


def count_words(words: list[str], files: list[Path]) -> tuple[dict[str, int], int, str]:
    """返回 (word -> 语料出现次数, 语料总字数, 计数后端)。缓存命中则不重扫。"""
    signature = corpus_signature(files)
    cache = read_cache(signature)
    cached: dict[str, int] = dict(cache["counts"]) if cache else {}
    corpus_chars = int(cache["corpus_chars"]) if cache else 0
    backend = str(cache["backend"]) if cache else ""

    missing = [w for w in words if w not in cached]
    log(f"[语料] 候选 {len(words)} 条，缓存命中 {len(words) - len(missing)} 条，待扫 {len(missing)} 条")
    if missing:
        try:
            import ahocorasick  # noqa: F401

            backend = "pyahocorasick"
            scanner = _scan_ahocorasick
        except ImportError:
            backend = "pure-python-4gram"
            log("[语料] 未安装 pyahocorasick，退回纯 Python 滑动窗口（慢数倍）")
            scanner = _scan_pure_python
        started = time.time()
        fresh = scanner(missing, files)
        counts = {word: fresh.get(idx, 0) for idx, word in enumerate(missing)}
        cached.update(counts)
        corpus_chars = sum(len(f.read_text(encoding="utf-8", errors="ignore")) for f in files)
        CACHE_PATH.write_text(
            json.dumps(
                {
                    "corpus_signature": signature,
                    "corpus_chars": corpus_chars,
                    "backend": backend,
                    "counts": cached,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        log(f"[语料] 新扫描 {len(missing)} 条用时 {time.time() - started:.1f}s，缓存写入 {CACHE_PATH}")
    else:
        log(f"[语料] 全部命中缓存 {CACHE_PATH}，未重扫语料（后端 {backend}）")

    return {w: int(cached[w]) for w in words}, corpus_chars, backend


# --------------------------------------------------------------------------- 过滤


def pick_min_count(survivors_by_threshold: dict[int, int]) -> tuple[int, str]:
    """选一个让最终条数落在 FINAL_COUNT_RANGE 的门槛：在合格门槛里取最大者（最严）。"""
    low, high = FINAL_COUNT_RANGE
    ordered = sorted(survivors_by_threshold)
    in_range = [th for th in ordered if low <= survivors_by_threshold[th] <= high]
    if in_range:
        chosen = max(in_range)
        next_higher = [th for th in ordered if th > chosen]
        higher_note = (
            f"再高一档 {min(next_higher)} 只剩 {survivors_by_threshold[min(next_higher)]} 条，跌破 {low}"
            if next_higher
            else f"已是最高门槛 {chosen}"
        )
        return chosen, (
            f"门槛 {chosen} 是使最终条数落在 {low}-{high} 内的最严（最大）门槛：{higher_note}；"
            f"再低一档（{max([th for th in ordered if th < chosen], default=chosen)}）会把更多不常用词放进来，"
            f"最终 {survivors_by_threshold[chosen]} 条"
        )
    best = min(ordered)
    return best, f"任何门槛都不到 {low} 条，取最低门槛 {best}，最终 {survivors_by_threshold[best]} 条，如实报告未达目标区间"


def build() -> dict:
    idioms = load_idioms(IDIOM_JSON)
    char_frequency = load_char_frequency(CHAR_BASE_JSON)
    common_chars = {c for c, f in char_frequency.items() if f <= MAX_CHAR_FREQUENCY}

    step = 0

    def report(label: str, before: int, after: int) -> None:
        nonlocal step
        step += 1
        log(f"[过滤 {step}] {label}: {before} -> {after}（淘汰 {before - after}）")

    # 1. 去重
    words = [rec["word"] for rec in idioms if isinstance(rec.get("word"), str)]
    unique = sorted(set(words))
    report("词典内去重", len(words), len(unique))

    # 2. 恰好 4 字
    length_ok = [w for w in unique if len(w) == WORD_LEN]
    report(f"长度恰好 {WORD_LEN} 字", len(unique), len(length_ok))

    # 3. 全部为汉字
    han_ok = [w for w in length_ok if HAN_RE.match(w)]
    report("全部由 \\u4e00-\\u9fff 组成", len(length_ok), len(han_ok))

    # 4. 全部字频 <= 2（3500 常用字）
    common_ok = [w for w in han_ok if all(ch in common_chars for ch in w)]
    report(f"四字 frequency<={MAX_CHAR_FREQUENCY}", len(han_ok), len(common_ok))

    # 5. 语料常用度
    counts, corpus_chars, backend = count_words(common_ok, corpus_files())
    survivors = {th: sum(1 for w in common_ok if counts[w] >= th) for th in THRESHOLD_CANDIDATES}
    log("[门槛] 各 MIN_COUNT 下剩余条数：")
    for th in THRESHOLD_CANDIDATES:
        log(f"         count >= {th:>3}: {survivors[th]}")

    min_count, rationale = pick_min_count(survivors)
    passed = [w for w in common_ok if counts[w] >= min_count]
    report(f"语料出现次数 >= {min_count}", len(common_ok), len(passed))

    # 6. 上限截断（按次数降序，次数相同按字典序，保证可重跑）
    ranked = sorted(passed, key=lambda w: (-counts[w], w))
    if len(ranked) > MAX_FINAL_COUNT:
        kept = ranked[:MAX_FINAL_COUNT]
        log(f"[截断] 超过上限 {MAX_FINAL_COUNT}，按语料次数取前 {MAX_FINAL_COUNT}（次数 {counts[kept[-1]]} 为分界）")
        ranked = kept
    final = sorted(ranked)

    return {
        "idioms_total": len(words),
        "words": final,
        "counts": counts,
        "corpus_chars": corpus_chars,
        "backend": backend,
        "survivors": survivors,
        "min_count": min_count,
        "rationale": rationale,
        "common_chars": sorted(common_chars),
    }


# --------------------------------------------------------------------------- 生成模块


def _wrap_chars(chars: list[str], width: int = 100) -> str:
    lines = []
    for i in range(0, len(chars), width):
        lines.append('    "' + "".join(chars[i : i + width]) + '"')
    return "\n".join(lines)


def _wrap_words(words: list[str]) -> str:
    return "\n".join(f'    "{word}",' for word in words)


def render(result: dict) -> str:
    counts = result["counts"]
    words = result["words"]
    top = sorted(words, key=lambda w: (-counts[w], w))[:10]
    provenance = {
        "source_repo": SOURCE_REPO,
        "source_commit": SOURCE_COMMIT,
        "source_license": SOURCE_LICENSE,
        "source_files": ["idiom/idiom.json", "character/char_base.json"],
        "source_fields": ["idiom.word", "character.char", "character.frequency"],
        "corpus_files": [f.name for f in corpus_files()],
        "corpus_chars": result["corpus_chars"],
        "count_backend": result["backend"],
        "count_cache": str(CACHE_PATH),
        "filters": [
            f"词条长度恰好 {WORD_LEN} 字",
            "四字全部落在 \\u4e00-\\u9fff",
            f"每个字的 char_base.frequency <= {MAX_CHAR_FREQUENCY}（全由 3500 常用字构成，共 {len(result['common_chars'])} 字）",
            "词典内按 word 去重",
            f"语料出现次数 >= {result['min_count']}（子串匹配计数，允许重叠）",
            f"超过 {MAX_FINAL_COUNT} 条时按语料次数降序取前 {MAX_FINAL_COUNT}",
        ],
        "thresholds": {
            "word_len": WORD_LEN,
            "max_char_frequency": MAX_CHAR_FREQUENCY,
            "min_count": result["min_count"],
            "max_final_count": MAX_FINAL_COUNT,
            "survivors_by_min_count": {str(k): v for k, v in result["survivors"].items()},
        },
        "min_count_rationale": result["rationale"],
        "final_count": len(words),
        "top10_by_count": {w: counts[w] for w in top},
    }
    provenance_literal = json.dumps(provenance, ensure_ascii=False, indent=4)
    return f'''"""常用成语白名单（由 scripts/build_idiom_lexicon.py 从开源词典生成）。

来源：{SOURCE_REPO} ({SOURCE_LICENSE})，commit {SOURCE_COMMIT}
生成方式：见 PROVENANCE；不要手改本文件，改生成脚本后重跑。
"""
from __future__ import annotations

import re

PROVENANCE: dict = {provenance_literal}

# char_base.frequency <= {MAX_CHAR_FREQUENCY} 的全部汉字（3397 个），validate() 用它做「常用字」兜底检查。
_COMMON_CHARS: str = (
{_wrap_chars(result["common_chars"])}
)

# 原始顺序列表：保留重复项，validate() 用它检查是否被 frozenset 静默吞掉重复。
_RAW_IDIOMS: tuple[str, ...] = (
{_wrap_words(words)}
)

IDIOMS: frozenset[str] = frozenset(_RAW_IDIOMS)


def validate() -> list[str]:
    """返回全部违规项；空列表 = 通过。"""
    problems: list[str] = []
    han = re.compile(r"^[\\u4e00-\\u9fff]+$")

    for word in sorted(set(_RAW_IDIOMS) | IDIOMS):
        if len(word) != {WORD_LEN}:
            problems.append(f"长度不是 {WORD_LEN} 字: {{word!r}}")
        elif not han.match(word):
            problems.append(f"含非汉字字符: {{word!r}}")

    if len(IDIOMS) != len(_RAW_IDIOMS):
        problems.append(f"存在重复词条: IDIOMS={{len(IDIOMS)}} 条，_RAW_IDIOMS={{len(_RAW_IDIOMS)}} 条")

    if not IDIOMS:
        problems.append("IDIOMS 为空")
    elif len(IDIOMS) < 500:
        problems.append(f"IDIOMS 仅 {{len(IDIOMS)}} 条，少于下限 500")

    common = set(_COMMON_CHARS)
    rare = [f"{{word}}:{{ch}}" for word in IDIOMS for ch in word if ch not in common]
    if rare:
        problems.append(f"含 frequency > {MAX_CHAR_FREQUENCY} 的字 {{len(rare)}} 处，例如 {{rare[:10]}}")

    print(
        f"[idiom_lexicon] {{len(IDIOMS)}} 条成语；"
        f"4 字纯中文 / 无重复 / 条数 >= 500 / 常用字频率 <= {MAX_CHAR_FREQUENCY} 四项检查完成；"
        f"违规 {{len(problems)}} 项"
    )
    return problems
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="只打印各步淘汰数，不写文件")
    args = parser.parse_args()

    result = build()
    log(f"[结果] 最终 {len(result['words'])} 条，语料 {result['corpus_chars']} 字，计数后端 {result['backend']}")
    if args.dry_run:
        return 0

    source = render(result)
    OUT_PATH.write_text(source, encoding="utf-8")
    OUT_PATH.chmod(0o644)
    log(f"[写出] {OUT_PATH}（{len(source)} 字节）")

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from dtseek.tasks.builtin.idiom import lexicon

    problems = lexicon.validate()
    log(f"[自检] validate() -> {problems}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
