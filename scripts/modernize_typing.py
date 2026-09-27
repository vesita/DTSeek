#!/usr/bin/env python3
"""把 typing 的 Deprecated 泛型别名替换成 PEP 585/604 内建写法。

    Dict[K, V]      -> dict[K, V]
    List[X]         -> list[X]
    Tuple[...]      -> tuple[...]
    Set[X]          -> set[X]
    FrozenSet[X]    -> frozenset[X]
    Optional[X]     -> X | None       （含嵌套，靠括号配对）
    Union[A, B, C]  -> A | B | C      （含嵌套，靠括号配对）

之后清理 `from typing import ...` 里已不再使用的名字；整行不再需要时连同换行删除
（注意保留 PEP 8 的类/函数前两空行，不做全局空行压缩）。

项目 requires-python >= 3.12，PEP 585 / 604 完全可用，无需 `from __future__`。

用法：
    uv run python scripts/modernize_typing.py            # 应用
    uv run python scripts/modernize_typing.py --check    # 只报告不写入
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # 项目根（scripts/ 的上一级）
TARGET_DIRS = ["src", "training", "examples", "tests", "scripts"]
SELF = Path(__file__).resolve()

SIMPLE = {
    "Dict": "dict",
    "List": "list",
    "Tuple": "tuple",
    "Set": "set",
    "FrozenSet": "frozenset",
}
# 这些依然从 typing 导入（未被 PEP 585/604 取代）
KEEP = {"Any", "Callable", "Iterable", "Sequence", "Awaitable", "Protocol", "TYPE_CHECKING"}


def _match_bracket(text: str, open_idx: int) -> int:
    """返回与 text[open_idx] == '[' 配对的 ']' 下标；找不到返回 -1。"""
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "[":
            depth += 1
        elif text[i] == "]":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _split_top_level(s: str) -> list:
    """按顶层逗号切分（忽略嵌套括号内的逗号）。"""
    parts, depth, cur = [], 0, []
    for ch in s:
        if ch in "[(":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append("".join(cur).strip())
    return [p for p in parts if p]


def _replace_generic(text: str, name: str, render) -> str:
    """把 `name[...]` 逐个替换成 render(inner) 的结果。"""
    out, i = [], 0
    while True:
        m = re.search(rf"\b{name}\[", text[i:])
        if not m:
            out.append(text[i:])
            break
        start = i + m.start()
        open_idx = i + m.end() - 1
        close_idx = _match_bracket(text, open_idx)
        if close_idx == -1:
            out.append(text[i:])
            break
        inner = text[open_idx + 1:close_idx]
        out.append(text[i:start])
        out.append(render(inner))
        i = close_idx + 1
    return "".join(out)


def rewrite(text: str) -> str:
    # Optional[X] -> X | None（内层若已是 union，才需要括号）
    def _render_optional(inner: str) -> str:
        expr = rewrite(inner)
        if "|" in expr:
            expr = f"({expr})"
        return f"{expr} | None"

    text = _replace_generic(text, "Optional", _render_optional)

    # Union[A, B, C] -> A | B | C
    text = _replace_generic(
        text, "Union",
        lambda inner: " | ".join(rewrite(p) for p in _split_top_level(inner)),
    )

    # 简单别名：先带方括号的（Dict[K,V]），再兜底裸引用（List[Dict] 里的内层 Dict）
    # —— 漏掉第二遍会让 `List[Dict]` 变成 `list[Dict]`，随后 import 被清理 ⇒ NameError。
    for cap, low in SIMPLE.items():
        text = re.sub(rf"\b{cap}\[", f"{low}[", text)
    for cap, low in SIMPLE.items():
        text = re.sub(rf"\b{cap}\b", low, text)

    # 清理 from typing import 行（保留换行，避免吃掉后续空行）
    def _clean_import(m):
        names = [n.strip() for n in m.group(1).split(",") if n.strip()]
        kept = [n for n in names if n in KEEP]
        # 整行删除时连换行一起去掉；保留时把换行原样带回来
        return f"from typing import {', '.join(kept)}\n" if kept else ""

    text = re.sub(r"from typing import ([^\n]+)\n", _clean_import, text)
    return text


def main():
    check_only = "--check" in sys.argv
    changed = []
    for d in TARGET_DIRS:
        for path in sorted((ROOT / d).rglob("*.py")):
            if path.resolve() == SELF:      # 不改自己，避免自我改写弄乱注释
                continue
            src = path.read_text(encoding="utf-8")
            new = rewrite(src)
            if new != src:
                changed.append(path.relative_to(ROOT))
                if not check_only:
                    path.write_text(new, encoding="utf-8")

    for p in changed:
        print(("would rewrite " if check_only else "rewrote ") + str(p))
    print(f"\n{'待修改' if check_only else '共修改'} {len(changed)} 个文件")


if __name__ == "__main__":
    sys.exit(main())
