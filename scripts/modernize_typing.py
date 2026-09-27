#!/usr/bin/env python3
"""把 typing 的 Deprecated 泛型别名替换成 PEP 585/604 内建写法。

  Dict[K, V]      -> dict[K, V]
  List[X]         -> list[X]
  Tuple[...]      -> tuple[...]
  Set[X]          -> set[X]
  FrozenSet[X]    -> frozenset[X]
  Optional[X]     -> X | None        （含嵌套，靠括号配对）
  Union[A, B, C]  -> A | B | C       （含嵌套，靠括号配对）

之后清理 `from typing import ...` 里已不再使用的名字；若整行空了就删掉该行。
项目 requires-python >= 3.12，PEP 585 / 604 完全可用，不需要 `from __future__`。
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # 项目根（scripts/ 的上一级）
TARGET_DIRS = ["src", "training", "examples", "tests", "scripts"]

SIMPLE = {
    "Dict": "dict",
    "List": "list",
    "Tuple": "tuple",
    "Set": "set",
    "FrozenSet": "frozenset",
}
# 这些依然从 typing 导入（没有被 PEP 585/604 取代）
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


def rewrite(text: str) -> str:
    # 1. Optional[X] -> X | None（从右往左扫，避免下标漂移）
    out, i = [], 0
    while True:
        m = re.search(r"\bOptional\[", text[i:])
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
        # 内层可能还有 Optional/Union，递归处理
        out.append(f"({rewrite(inner)}) | None")
        i = close_idx + 1
    text = "".join(out)

    # 2. Union[A, B, C] -> A | B | C
    out, i = [], 0
    while True:
        m = re.search(r"\bUnion\[", text[i:])
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
        out.append(" | ".join(rewrite(p) for p in _split_top_level(inner)))
        i = close_idx + 1
    text = "".join(out)

    # 3. 简单别名 Dict/List/Tuple/Set/FrozenSet
    for cap, low in SIMPLE.items():
        text = re.sub(rf"\b{cap}\[", f"{low}[", text)

    # 4. 清理 from typing import 行
    def _clean_import(m):
        names = [n.strip() for n in m.group(1).split(",") if n.strip()]
        kept = [n for n in names if n in KEEP]
        return f"from typing import {', '.join(kept)}" if kept else ""

    text = re.sub(r"from typing import ([^\n]+)", _clean_import, text)
    # 删除因清理而空掉的行（以及它留下的空行）
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def main():
    changed = []
    for d in TARGET_DIRS:
        for path in sorted((ROOT / d).rglob("*.py")):
            src = path.read_text(encoding="utf-8")
            new = rewrite(src)
            if new != src:
                path.write_text(new, encoding="utf-8")
                changed.append(path.relative_to(ROOT))
    for p in changed:
        print("rewrote", p)
    print(f"\n共修改 {len(changed)} 个文件")


if __name__ == "__main__":
    sys.exit(main())
